"""
A script for training a diffusion model to unconditional image generation.
"""

import argparse
import numpy as np
import random
import sys
import torch as th

sys.path.append(".")
sys.path.append("..")

from guided_diffusion import (dist_util,
                              logger)
from guided_diffusion.bratsloader import BRATSVolumes
from guided_diffusion.lidcloader import LIDCVolumes
from guided_diffusion.inpaintloader import InpaintVolumes
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (model_and_diffusion_defaults,
                                          create_model_and_diffusion,
                                          args_to_dict,
                                          add_dict_to_argparser)
from guided_diffusion.train_util import TrainLoop
from guided_diffusion.pretrain_checks import run_pretrain_checks
from torch.utils.tensorboard import SummaryWriter


def run_training(args):
    seed = args.seed
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    summary_writer = None
    if args.use_tensorboard:
        logdir = None
        if args.tensorboard_path:
            logdir = args.tensorboard_path
        summary_writer = SummaryWriter(log_dir=logdir)
        summary_writer.add_text(
            'config',
            '\n'.join([f'--{k}={repr(v)} <br/>' for k, v in vars(args).items()])
        )
        logger.configure(dir=summary_writer.get_logdir())
    else:
        logger.configure()

    dist_util.setup_dist(devices=args.devices)

    logger.log("Creating model and diffusion...")
    arguments = args_to_dict(args, model_and_diffusion_defaults().keys())
    model, diffusion = create_model_and_diffusion(**arguments)

    model.to(dist_util.dev([0, 1]) if len(args.devices) > 1 else dist_util.dev())  # allow for 2 devices
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=1000)

    dataset_size = args.dataset_image_size or args.image_size
    if args.dataset == 'brats':
        assert args.image_size in [128, 256], "We currently just support image sizes: 128, 256"
        ds = BRATSVolumes(
            args.data_dir,
            test_flag=False,
            normalize=(lambda x: 2*x - 1) if args.renormalize else None,
            mode='train',
            img_size=dataset_size,
            cache=args.cache_dataset,
        )
        val_loader = None

    elif args.dataset == 'lidc-idri':
        assert args.image_size in [128, 256], "We currently just support image sizes: 128, 256"
        ds = LIDCVolumes(
            args.data_dir,
            test_flag=False,
            normalize=(lambda x: 2 * x - 1) if args.renormalize else None,
            mode='train',
            img_size=dataset_size,
        )
        val_loader = None

    elif args.dataset == 'inpaint':
        ds = InpaintVolumes(
            args.data_dir,
            subset='train',
            img_size=dataset_size,
            desired_image_size=args.desired_image_size,
            normalize=(lambda x: 2 * x - 1) if args.renormalize else None,
            cache=args.cache_dataset,
        )
        val_ds = InpaintVolumes(
            args.data_dir,
            subset='val',
            img_size=dataset_size,
            desired_image_size=args.desired_image_size,
            normalize=(lambda x: 2 * x - 1) if args.renormalize else None,
            cache=args.cache_dataset,
        )
        val_loader = th.utils.data.DataLoader(
            val_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
    else:
        print("We currently just support the datasets: brats, lidc-idri, inpaint")
        val_loader = None

    datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        persistent_workers=args.num_workers > 0,
    )

    if args.run_tests:
        logger.log("Running pre-training checks...")
        run_pretrain_checks(args, datal, model, diffusion, schedule_sampler)
        return 0.0

    logger.log("Start training...")
    return TrainLoop(
        model=model,
        diffusion=diffusion,
        data=datal,
        batch_size=args.batch_size,
        in_channels=args.in_channels,
        image_size=args.image_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        resume_step=args.resume_step,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        dataset=args.dataset,
        summary_writer=summary_writer,
        mode='inpaint' if args.dataset == 'inpaint' else 'default',
        val_data=val_loader,
        val_interval=args.val_interval,
        early_stop=args.early_stop,
        patience=args.patience,
        min_delta=args.min_delta,
    ).run_loop()


def main():
    args = create_argparser().parse_args()
    if args.optuna_trials > 0:
        import optuna

        def objective(trial):
            trial_args = argparse.Namespace(**vars(args))
            trial_args.lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
            trial_args.batch_size = trial.suggest_categorical(
                "batch_size", [1, 2, 4, 8]
            )
            trial_args.dropout = trial.suggest_float("dropout", 0.0, 0.5)
            return run_training(trial_args)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=args.optuna_trials)
        print("Best hyperparameters:", study.best_params)
    else:
        run_training(args)


def create_argparser():
    defaults = dict(
        seed=0,
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=500,
        save_interval=5000,
        resume_checkpoint='',
        resume_step=0,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        dataset='brats',
        use_tensorboard=True,
        tensorboard_path='',  # set path to existing logdir for resuming
        devices=[0],
        dims=3,
        learn_sigma=False,
        num_groups=32,
        channel_mult="1,2,2,4,4",
        in_channels=8,
        out_channels=8,
        bottleneck_attention=False,
        num_workers=0,
        mode='default',
        renormalize=True,
        additive_skips=False,
        use_freq=False,
        val_interval=1000,
        run_tests=False,
        cache_dataset=True,
        early_stop=False,
        patience=10,
        min_delta=0.0,
        optuna_trials=0,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument("--desired_image_size", type=int, default=None)
    parser.add_argument("--dataset_image_size", type=int, default=None)
    return parser


if __name__ == "__main__":
    main()
