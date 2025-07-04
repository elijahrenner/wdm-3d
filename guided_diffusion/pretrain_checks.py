import os
import json
import torch as th

from . import dist_util, logger


def run_pretrain_checks(args, dataloader, model, diffusion, schedule_sampler):
    """Run a set of quick checks before starting training."""
    logdir = logger.get_dir() or os.getcwd()
    os.makedirs(logdir, exist_ok=True)

    # Save hyperparameters
    with open(os.path.join(logdir, "hyperparameters.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Fetch one batch from dataloader and save example inputs
    sample = next(iter(dataloader))
    if args.dataset == "inpaint":
        example = sample[0]
    else:
        example = sample
    expected_shape = (args.image_size, args.image_size, args.image_size)
    actual_shape = tuple(example.shape[-3:])
    if actual_shape != expected_shape:
        logger.log(
            f"Warning: example volume shape {actual_shape} does not match image_size {expected_shape}"
        )
    th.save(example.cpu(), os.path.join(logdir, "example_input.pt"))

    model.to(dist_util.dev([0, 1]) if len(args.devices) > 1 else dist_util.dev())
    model.train()

    cond = {}
    if args.dataset == "inpaint":
        cond = {"mask": sample[1].to(dist_util.dev())}
        batch = sample[0].to(dist_util.dev())
    else:
        batch = sample.to(dist_util.dev())

    def to_cpu(obj):
        if th.is_tensor(obj):
            return obj.cpu()
        if isinstance(obj, (list, tuple)):
            return [to_cpu(o) for o in obj]
        if isinstance(obj, dict):
            return {k: to_cpu(v) for k, v in obj.items()}
        return obj

    t, _ = schedule_sampler.sample(1, dist_util.dev())
    losses, _, _ = diffusion.training_losses(
        model,
        x_start=batch[:1],
        t=t,
        model_kwargs=cond,
        labels=None,
        mode="inpaint" if args.dataset == "inpaint" else "default",
    )

    loss = losses["mse_wav"].mean()
    if not th.isfinite(loss):
        raise RuntimeError("Non-finite loss encountered during checks")

    loss.backward()
    for p in model.parameters():
        if p.grad is None:
            continue
        if not th.isfinite(p.grad).all():
            raise RuntimeError("Non-finite gradient encountered during checks")

    logger.log("Pre-training checks passed.")
