import copy
import functools
import os
import nibabel as nib
import numpy as np

import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.utils.tensorboard
from torch.optim import AdamW
import torch.amp as amp

import itertools

from . import dist_util, logger
from .resample import LossAwareSampler, UniformSampler
from DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D


def psnr(pred: th.Tensor, target: th.Tensor, data_range: float = 2.0) -> th.Tensor:
    """Compute the peak signal to noise ratio."""
    mse = th.mean((pred - target) ** 2)
    mse = th.clamp(mse, min=1e-8)
    return 20 * th.log10(th.tensor(data_range, device=pred.device)) - 10 * th.log10(mse)


def dice_score(pred: th.Tensor, target: th.Tensor, threshold: float = 0.0) -> th.Tensor:
    """Dice score for binary volumes."""
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()
    inter = (pred_bin * target_bin).sum()
    return 2 * inter / (pred_bin.sum() + target_bin.sum() + 1e-8)

INITIAL_LOG_LOSS_SCALE = 20.0

def visualize(img: th.Tensor) -> th.Tensor:
    """Normalize a tensor image to the range [0, 1]."""
    _min = img.min()
    _max = img.max()
    if _max == _min:
        # avoid division by zero which leads to NaNs in TensorBoard
        return th.zeros_like(img)
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        in_channels,
        image_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        resume_step,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        dataset='brats',
        val_data=None,
        val_interval=0,
        summary_writer=None,
        mode='default',
        loss_level='image',
        early_stop=False,
        patience=10,
        min_delta=0.0,
    ):
        self.summary_writer = summary_writer
        self.mode = mode
        self.model = model
        self.diffusion = diffusion
        self.datal = data
        self.dataset = dataset
        self.iterdatal = iter(data)
        self.val_data = val_data
        self.iterval = iter(val_data) if val_data is not None else None
        self.val_interval = val_interval
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.image_size = image_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        if self.use_fp16:
            self.grad_scaler = amp.GradScaler('cuda')
        else:
            self.grad_scaler = amp.GradScaler('cuda',enabled=False)

        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.dwt = DWT_3D('haar')
        self.idwt = IDWT_3D('haar')

        self.loss_level = loss_level

        self.early_stop = early_stop
        self.patience = patience
        self.min_delta = min_delta
        self.best_val = float('inf')
        self.no_improve = 0

        self.step = 1
        self.resume_step = resume_step
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()

        self.opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            print("Resume Step: " + str(self.resume_step))
            self._load_optimizer_state()

        if not th.cuda.is_available():
            logger.warn(
                "Training requires CUDA. "
            )

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            print('resume model ...')
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())


    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)
        else:
            print('no optimizer checkpoint exists')

    
    def _save_initial_sample(self):
        """Save an initial output sample before training starts."""
        if dist.get_rank() != 0:
            return

        sample = next(iter(self.datal))
        if self.dataset == 'inpaint':
            batch = sample[0].to(dist_util.dev())
            cond = {"mask": sample[1].to(dist_util.dev())}
        else:
            batch = sample.to(dist_util.dev())
            cond = {}

        t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())
        with th.no_grad():
            _, _, sample_idwt = self.diffusion.training_losses(
                self.model,
                x_start=batch,
                t=t,
                model_kwargs=cond,
                labels=None,
                mode=self.mode,
            )

        sample_out = (sample_idwt + 1) / 2.0
        out_dir = os.path.join(logger.get_dir(), "samples")
        os.makedirs(out_dir, exist_ok=True)
        nii_path = os.path.join(out_dir, "init_sample.nii.gz")
        pt_path = os.path.join(out_dir, "init_sample.pt")
        nib.save(nib.Nifti1Image(sample_out[0, 0].cpu().numpy(), np.eye(4)), nii_path)
        th.save(sample_out.cpu(), pt_path)
        print(f"Saved initial sample to {nii_path} and {pt_path}")

    def run_loop(self):
        import time
        self._save_initial_sample()
        print("Entering training loop...", flush=True)
        t = time.time()
        while not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps:
            t_total = time.time() - t
            t = time.time()
            if self.dataset in ['brats', 'lidc-idri', 'inpaint']:
                try:
                    batch = next(self.iterdatal)
                    cond = {}
                except StopIteration:
                    self.iterdatal = iter(self.datal)
                    batch = next(self.iterdatal)
                    cond = {}

            if self.dataset == 'inpaint':
                # Inpainting loader returns tuple
                batch = tuple(b.to(dist_util.dev()) if th.is_tensor(b) else b for b in batch)
            else:
                batch = batch.to(dist_util.dev())

            t_fwd = time.time()
            t_load = t_fwd-t

            lossmse, sample, sample_idwt = self.run_step(batch, cond)

            t_fwd = time.time()-t_fwd
            print(
                f"step {self.step} | load {t_load:.3f}s | forward {t_fwd:.3f}s | loss {lossmse.item():.6f}",
                flush=True,
            )

            names = ["LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH"]

            if self.summary_writer is not None:
                self.summary_writer.add_scalar('time/load', t_load, global_step=self.step + self.resume_step)
                self.summary_writer.add_scalar('time/forward', t_fwd, global_step=self.step + self.resume_step)
                self.summary_writer.add_scalar('time/total', t_total, global_step=self.step + self.resume_step)
                self.summary_writer.add_scalar('loss/MSE', lossmse.item(), global_step=self.step + self.resume_step)

            if self.step % 200 == 0:
                image_size = sample_idwt.size()[2]
                midplane = sample_idwt[0, 0, :, :, image_size // 2].detach().cpu()
                midplane = visualize(midplane)
                self.summary_writer.add_image(
                    'sample/x_0', midplane.unsqueeze(0),
                    global_step=self.step + self.resume_step
                )

                image_size = sample.size()[2]
                for ch in range(8):
                    midplane = sample[0, ch, :, :, image_size // 2].detach().cpu()
                    midplane = visualize(midplane)
                    self.summary_writer.add_image(
                        f'sample/{names[ch]}', midplane.unsqueeze(0),
                        global_step=self.step + self.resume_step
                    )

            if self.step % self.log_interval == 0:
                logger.dumpkvs()

            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return

            if (
                self.val_data is not None
                and self.val_interval > 0
                and self.step % self.val_interval == 0
            ):
                val = self._run_validation()
                if val is not None:
                    if val + self.min_delta < self.best_val:
                        self.best_val = val
                        self.no_improve = 0
                    else:
                        self.no_improve += 1
                        if self.early_stop and self.no_improve >= self.patience:
                            print("Early stopping triggered.", flush=True)
                            break
            self.step += 1

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
        return self.best_val

    def run_step(self, batch, cond, label=None, info=dict()):
        lossmse, sample, sample_idwt = self.forward_backward(batch, cond, label)

        if self.use_fp16:
            self.grad_scaler.unscale_(self.opt)  # check self.grad_scaler._per_optimizer_states

        # compute norms
        with torch.no_grad():
            param_max_norm = max([p.abs().max().item() for p in self.model.parameters()])
            grad_max_norm = max([p.grad.abs().max().item() for p in self.model.parameters()])
            info['norm/param_max'] = param_max_norm
            info['norm/grad_max'] = grad_max_norm

        if not torch.isfinite(lossmse): #infinite
            if not torch.isfinite(torch.tensor(param_max_norm)):
                logger.log(f"Model parameters contain non-finite value {param_max_norm}, entering breakpoint", level=logger.ERROR)
                breakpoint()
            else:
                logger.log(f"Model parameters are finite, but loss is not: {lossmse}"
                           "\n -> update will be skipped in grad_scaler.step()", level=logger.WARN)

        if self.use_fp16:
            print("Use fp16 ...")
            self.grad_scaler.step(self.opt)
            self.grad_scaler.update()
            info['scale'] = self.grad_scaler.get_scale()
        else:
            self.opt.step()
        self._anneal_lr()
        self.log_step()
        return lossmse, sample, sample_idwt

    @th.no_grad()
    def _run_validation(self):
        """Run a single validation pass."""
        if self.val_data is None:
            return

        self.model.eval()
        try:
            batch = next(self.iterval)
        except StopIteration:
            self.iterval = iter(self.val_data)
            batch = next(self.iterval)

        if self.dataset == 'inpaint':
            batch = tuple(b.to(dist_util.dev()) if th.is_tensor(b) else b for b in batch)
            cond = {"mask": batch[1]}
            gt = batch[0]
            mask = batch[1]
        else:
            batch = batch.to(dist_util.dev())
            cond = {}
            gt = batch
            mask = th.ones_like(gt)

        t, _ = self.schedule_sampler.sample(gt.shape[0], dist_util.dev())
        loss_dict, sample_wav, sample_idwt = self.diffusion.training_losses(
            self.model,
            x_start=gt,
            t=t,
            model_kwargs=cond,
            labels=None,
            mode=self.mode,
        )

        weights = th.ones(len(loss_dict["mse_wav"])).to(sample_idwt.device)
        val_loss = (loss_dict["mse_wav"] * weights).mean()
        val_loss = th.clamp(val_loss, min=0).item()

        pred_region = sample_idwt * mask
        gt_region = gt * mask

        val_psnr = psnr(pred_region, gt_region).item()
        val_dice = dice_score(pred_region, gt_region).item()

        if self.summary_writer is not None:
            self.summary_writer.add_scalar(
                "val/loss", val_loss, global_step=self.step + self.resume_step
            )
            self.summary_writer.add_scalar(
                "val/psnr", val_psnr, global_step=self.step + self.resume_step
            )
            self.summary_writer.add_scalar(
                "val/dice", val_dice, global_step=self.step + self.resume_step
            )

            mid = pred_region.shape[-1] // 2
            gt_slice = visualize(gt_region[0, 0, :, :, mid])
            pred_slice = visualize(pred_region[0, 0, :, :, mid])
            viz = th.cat([gt_slice, pred_slice], dim=-1)
            self.summary_writer.add_image(
                "val/inpaint_vs_gt",
                viz.unsqueeze(0),
                global_step=self.step + self.resume_step,
            )
        print(f"val_loss {val_loss:.6f} | PSNR {val_psnr:.3f} | Dice {val_dice:.3f}", flush=True,)

        self.model.train()
        return val_loss

    def forward_backward(self, batch, cond, label=None):
        for p in self.model.parameters():  # Zero out gradient
            p.grad = None

        for i in range(0, batch[0].shape[0] if self.dataset == 'inpaint' else batch.shape[0], self.microbatch):
            if self.dataset == 'inpaint':
                micro = batch[0][i: i + self.microbatch]
                micro_mask = batch[1][i: i + self.microbatch]
                micro_cond = {"mask": micro_mask}
            else:
                micro = batch[i: i + self.microbatch]
                micro_cond = None
            micro = micro.to(dist_util.dev())
            if self.dataset == 'inpaint':
                micro_cond = {k: v.to(dist_util.dev()) for k, v in micro_cond.items()}

            if label is not None:
                micro_label = label[i: i + self.microbatch].to(dist_util.dev())
            else:
                micro_label = None

            last_batch = (
                i + self.microbatch
            ) >= (batch[0].shape[0] if self.dataset == 'inpaint' else batch.shape[0])
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(self.diffusion.training_losses,
                                               self.model,
                                               x_start=micro,
                                               t=t,
                                               model_kwargs=micro_cond,
                                               labels=micro_label,
                                               mode=self.mode,
                                               )
            losses1 = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses1["loss"].detach()
                )

            losses = losses1[0]         # Loss value
            sample = losses1[1]         # Denoised subbands at t=0
            sample_idwt = losses1[2]    # Inverse wavelet transformed denoised subbands at t=0

            # Log wavelet level loss
            self.summary_writer.add_scalar('loss/mse_wav_lll', losses["mse_wav"][0].item(),
                                           global_step=self.step + self.resume_step)
            self.summary_writer.add_scalar('loss/mse_wav_llh', losses["mse_wav"][1].item(),
                                           global_step=self.step + self.resume_step)
            self.summary_writer.add_scalar('loss/mse_wav_lhl', losses["mse_wav"][2].item(),
                                           global_step=self.step + self.resume_step)
            self.summary_writer.add_scalar('loss/mse_wav_lhh', losses["mse_wav"][3].item(),
                                           global_step=self.step + self.resume_step)
            self.summary_writer.add_scalar('loss/mse_wav_hll', losses["mse_wav"][4].item(),
                                           global_step=self.step + self.resume_step)
            self.summary_writer.add_scalar('loss/mse_wav_hlh', losses["mse_wav"][5].item(),
                                           global_step=self.step + self.resume_step)
            self.summary_writer.add_scalar('loss/mse_wav_hhl', losses["mse_wav"][6].item(),
                                           global_step=self.step + self.resume_step)
            self.summary_writer.add_scalar('loss/mse_wav_hhh', losses["mse_wav"][7].item(),
                                           global_step=self.step + self.resume_step)

            weights = th.ones(len(losses["mse_wav"])).cuda()  # Equally weight all wavelet channel losses

            # Clamp per-channel losses to avoid negative values from numeric errors
            losses["mse_wav"] = th.clamp(losses["mse_wav"], min=0)
            loss = (losses["mse_wav"] * weights).mean()
            loss = th.clamp(loss, min=0)
            lossmse = loss.detach()

            log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})

            # perform some finiteness checks
            if not torch.isfinite(loss):
                logger.log(f"Encountered non-finite loss {loss}")
            if self.use_fp16:
                self.grad_scaler.scale(loss).backward()
            else:
                loss.backward()

            return lossmse.detach(), sample, sample_idwt

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, state_dict):
            if dist.get_rank() == 0:
                logger.log("Saving model...")
                if self.dataset == 'brats':
                    filename = f"brats_{(self.step+self.resume_step):06d}.pt"
                elif self.dataset == 'lidc-idri':
                    filename = f"lidc-idri_{(self.step+self.resume_step):06d}.pt"
                elif self.dataset == 'inpaint':
                    filename = f"inpaint_{(self.step+self.resume_step):06d}.pt"
                else:
                    raise ValueError(f'dataset {self.dataset} not implemented')

                with bf.BlobFile(bf.join(get_blob_logdir(), 'checkpoints', filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.model.state_dict())

        if dist.get_rank() == 0:
            checkpoint_dir = os.path.join(logger.get_dir(), 'checkpoints')
            with bf.BlobFile(
                bf.join(checkpoint_dir, f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """

    split = os.path.basename(filename)
    split = split.split(".")[-2]  # remove extension
    split = split.split("_")[-1]  # remove possible underscores, keep only last word
    # extract trailing number
    reversed_split = []
    for c in reversed(split):
        if not c.isdigit():
            break
        reversed_split.append(c)
    split = ''.join(reversed(reversed_split))
    split = ''.join(c for c in split if c.isdigit())  # remove non-digits
    try:
        return int(split)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
