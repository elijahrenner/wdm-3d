import argparse
import random
import numpy as np
import torch as th
import optuna

from generation_train import create_argparser
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSVolumes
from guided_diffusion.lidcloader import LIDCVolumes
from guided_diffusion.inpaintloader import InpaintVolumes
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (model_and_diffusion_defaults,
                                          create_model_and_diffusion,
                                          args_to_dict)
from guided_diffusion.train_util import TrainLoop


def _get_dataloaders(args):
    dataset_size = args.dataset_image_size or args.image_size
    if args.dataset == 'brats':
        ds = BRATSVolumes(
            args.data_dir,
            test_flag=False,
            normalize=(lambda x: 2 * x - 1) if args.renormalize else None,
            mode='train',
            img_size=dataset_size,
            cache=args.cache_dataset,
        )
        val_loader = None
    elif args.dataset == 'lidc-idri':
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
        raise ValueError('dataset not supported')

    datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        persistent_workers=args.num_workers > 0,
    )
    return datal, val_loader


def objective(trial, base_args):
    args = argparse.Namespace(**vars(base_args))
    args.lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    args.dropout = trial.suggest_float('dropout', 0.0, 0.3)
    args.batch_size = trial.suggest_categorical('batch_size', [1, 2, 4, 8, 10])
    args.early_stopping = True
    args.early_stopping_patience = 5

    seed = args.seed
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dist_util.setup_dist(devices=args.devices)
    logger.configure()

    arguments = args_to_dict(args, model_and_diffusion_defaults().keys())
    model, diffusion = create_model_and_diffusion(**arguments)
    model.to(dist_util.dev([0, 1]) if len(args.devices) > 1 else dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion, maxt=1000)

    datal, val_loader = _get_dataloaders(args)

    loop = TrainLoop(
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
        resume_checkpoint='',
        resume_step=0,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        dataset=args.dataset,
        summary_writer=None,
        mode='inpaint' if args.dataset == 'inpaint' else 'default',
        val_data=val_loader,
        val_interval=args.val_interval,
        early_stopping=args.early_stopping,
        patience=args.early_stopping_patience,
    )
    loop.run_loop()
    return loop.best_val_loss


def main():
    parser = create_argparser()
    parser.add_argument('--n_trials', type=int, default=20)
    args = parser.parse_args()

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)
    print('Best trial:', study.best_trial.number)
    print('Best value:', study.best_trial.value)
    print('Best params:', study.best_trial.params)


if __name__ == '__main__':
    main()
