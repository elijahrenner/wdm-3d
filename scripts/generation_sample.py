"""
A script for sampling from a diffusion model for unconditional image generation.
"""

import argparse
import nibabel as nib
import numpy as np
import os
import pathlib
import random
import sys
import torch as th

sys.path.append(".")

from guided_diffusion import (dist_util,
                              logger)
from guided_diffusion.script_util import (model_and_diffusion_defaults,
                                          create_model_and_diffusion,
                                          add_dict_to_argparser,
                                          args_to_dict,
                                          )
from guided_diffusion.inpaintloader import InpaintVolumes
from DWT_IDWT.DWT_IDWT_layer import IDWT_3D, DWT_3D


def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img


def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()


def main():
    args = create_argparser().parse_args()
    seed = args.seed
    dist_util.setup_dist(devices=args.devices)
    logger.configure()

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    logger.log("Load model from: {}".format(args.model_path))
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev([0, 1]) if len(args.devices) > 1 else dist_util.dev())  # allow for 2 devices

    if args.use_fp16:
        raise ValueError("fp16 currently not implemented")

    model.eval()
    idwt = IDWT_3D("haar")
    dwt = DWT_3D("haar")

    dataset_size = args.dataset_image_size or args.image_size
    if args.dataset == 'inpaint':
        ds = InpaintVolumes(
            args.data_dir,
            subset='val',
            img_size=dataset_size,
            desired_image_size=args.desired_image_size,
        )
        loader = th.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        data_iter = iter(loader)
    else:
        loader = None

    for ind in range(args.num_samples // args.batch_size):
        th.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # print(f"Reseeded (in for loop) to {seed}")

        seed += 1

        if args.dataset == 'inpaint':
            try:
                Y, M, Y_void, name, affine = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                Y, M, Y_void, name, affine = next(data_iter)
            Y = Y.to(dist_util.dev())
            M = M.to(dist_util.dev())
            Y_void = Y_void.to(dist_util.dev())
            # Wavelet domain context and mask
            ctx = th.cat(dwt(Y_void), dim=1)
            mask_dwt = th.cat(dwt(M), dim=1)
            mask_rep = mask_dwt.repeat(1, Y.shape[1], 1, 1, 1)
            noise = th.randn_like(ctx)
            sample = diffusion.p_sample_loop(
                model,
                shape=ctx.shape,
                noise=noise,
                clip_denoised=args.clip_denoised,
                model_kwargs={'context': ctx, 'mask': mask_rep},
            )
        else:
            img = th.randn(
                args.batch_size,
                8,
                args.image_size // 2,
                args.image_size // 2,
                args.image_size // 2,
                device=dist_util.dev(),
            )
            sample = diffusion.p_sample_loop(
                model=model,
                shape=img.shape,
                noise=img,
                clip_denoised=args.clip_denoised,
                model_kwargs={},
            )

        B, _, D, H, W = sample.size()

        sample = idwt(sample[:, 0, :, :, :].view(B, 1, D, H, W) * 3.,
                      sample[:, 1, :, :, :].view(B, 1, D, H, W),
                      sample[:, 2, :, :, :].view(B, 1, D, H, W),
                      sample[:, 3, :, :, :].view(B, 1, D, H, W),
                      sample[:, 4, :, :, :].view(B, 1, D, H, W),
                      sample[:, 5, :, :, :].view(B, 1, D, H, W),
                      sample[:, 6, :, :, :].view(B, 1, D, H, W),
                      sample[:, 7, :, :, :].view(B, 1, D, H, W))

        sample = (sample + 1) / 2.


        if len(sample.shape) == 5:
            sample = sample.squeeze(dim=1)  # don't squeeze batch dimension for bs 1

        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        for i in range(sample.shape[0]):
            if args.dataset == 'inpaint':
                output_name = os.path.join(args.output_dir, f"{name[i]}_inpaint.nii.gz")
                aff = affine[i] if isinstance(affine[i], np.ndarray) else affine[i].numpy()
            else:
                output_name = os.path.join(args.output_dir, f'sample_{ind}_{i}.nii.gz')
                aff = np.eye(4)
            img = nib.Nifti1Image(sample.detach().cpu().numpy()[i, :, :, :], aff)
            nib.save(img=img, filename=output_name)
            print(f'Saved to {output_name}')


def create_argparser():
    defaults = dict(
        seed=0,
        data_dir="",
        data_mode='validation',
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        class_cond=False,
        sampling_steps=0,
        model_path="",
        devices=[0],
        output_dir='./results',
        mode='default',
        renormalize=False,
        image_size=256,
        desired_image_size=None,
        dataset_image_size=None,
        half_res_crop=False,
        concat_coords=False, # if true, add 3 (for 3d) or 2 (for 2d) to in_channels
    )
    defaults.update({k:v for k, v in model_and_diffusion_defaults().items() if k not in defaults})
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
