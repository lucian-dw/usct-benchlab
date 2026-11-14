"""Train a reference synthetic prior or an explicitly supplied speed-image set.

python -m usctbench.algorithms.fttt.train_prior --output checkpoints/fttt.pt
No observations or reconstruction ground truths are read by this training tool.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .prior import PriorMetadata, save_prior, file_sha256
from usctbench._vendor.inv_framework.models.tiny_unet import TinyUNet
from usctbench._vendor.inv_framework.solvers.diffusion.scheduler import VPSchedule


def synthetic_speed_images(count, shape, seed):
    """Independent random ellipses on a 1500 m/s background; [image,y,x]."""
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(
        np.linspace(-1, 1, shape[0]), np.linspace(-1, 1, shape[1]), indexing="ij"
    )
    images = np.full((count, *shape), 1500.0, dtype=np.float32)
    for image in images:
        for _ in range(int(rng.integers(1, 4))):
            cy, cx = rng.uniform(-0.5, 0.5, 2)
            a, b = rng.uniform(0.15, 0.45, 2)
            angle = rng.uniform(0, np.pi)
            x = (xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle)
            y = -(xx - cx) * np.sin(angle) + (yy - cy) * np.cos(angle)
            image[(x / a) ** 2 + (y / b) ** 2 < 1] = rng.uniform(1400, 1600)
    return images


def _hash_images(images):
    return hashlib.sha256(np.ascontiguousarray(images).tobytes()).hexdigest()


def train_prior(
    *,
    output,
    steps=1000,
    shape=(16, 16),
    seed=70,
    batch_size=16,
    learning_rate=0.001,
    training_timesteps=1000,
    base_channels=8,
    time_embedding_dim=32,
    device="cpu",
    images=None,
):
    if Path(output).exists():
        raise FileExistsError(f"refusing to replace checkpoint: {output}")
    if (
        steps < 1
        or batch_size < 1
        or learning_rate <= 0
        or not np.isfinite(learning_rate)
    ):
        raise ValueError("steps, batch_size and learning_rate must be positive")
    bounds = (1300.0, 1700.0)
    provenance = {"seed": seed}
    if images is None:
        train = synthetic_speed_images(512, shape, seed)
        validation = synthetic_speed_images(32, shape, seed + 1)
        provenance.update(
            source="procedural_ellipse_speed_images_v1",
            limitation="Synthetic reference prior only; not a validated tissue prior.",
            validation_seed=seed + 1,
        )
    else:
        with np.load(images, allow_pickle=False) as archive:
            train, validation = (
                archive["train_speed_mps"],
                archive["validation_speed_mps"],
            )
        provenance.update(
            source=str(Path(images).resolve()), source_sha256=file_sha256(images)
        )
    for value in (train, validation):
        if value.ndim != 3 or value.shape[1:] != tuple(shape) or len(value) < 1:
            raise ValueError(
                "training/validation arrays must be nonempty [image,ny,nx]"
            )
        if (
            not np.isfinite(value).all()
            or value.min() < bounds[0]
            or value.max() > bounds[1]
        ):
            raise ValueError("speed images must be finite and within [1300,1700] m/s")
    train_hashes = {_hash_images(image) for image in train}
    if any(_hash_images(image) in train_hashes for image in validation):
        raise ValueError("training and validation contain duplicate images")
    provenance.update(
        train_sha256=_hash_images(train),
        validation_sha256=_hash_images(validation),
        train_count=len(train),
        validation_count=len(validation),
    )
    metadata = PriorMetadata(
        image_shape=shape,
        sound_speed_bounds_mps=bounds,
        base_channels=base_channels,
        time_embedding_dim=time_embedding_dim,
        training_timesteps=training_timesteps,
        trained_steps=steps,
        training_data=provenance,
        validation={"status": "pending"},
    )
    target_device = torch.device(device)
    cuda_ids = [target_device.index or 0] if target_device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_ids):
        torch.random.default_generator.manual_seed(seed)
        if cuda_ids:
            with torch.cuda.device(target_device):
                torch.cuda.manual_seed(seed)
        lower, upper = 1 / bounds[1], 1 / bounds[0]

        def normalized(value):
            return torch.tensor(
                ((1 / value - lower) / (upper - lower))[:, None],
                dtype=torch.float32,
                device=target_device,
            )

        train_tensor, validation_tensor = normalized(train), normalized(validation)
        model = TinyUNet(1, base_channels, time_embedding_dim).to(target_device)
        schedule = VPSchedule(training_timesteps, device=target_device)
        # Fixed, independent validation corruption, identical before/after training.
        generator = torch.Generator(device=target_device).manual_seed(seed + 2)
        val_t = torch.randint(
            training_timesteps,
            (len(validation),),
            generator=generator,
            device=target_device,
        )
        val_noise = torch.randn(
            validation_tensor.shape, generator=generator, device=target_device
        )
        val_noisy = schedule.add_noise(validation_tensor, val_noise, val_t)

        def score():
            with torch.no_grad():
                return float(F.mse_loss(model(val_noisy, val_t), val_noise))

        initial = score()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        history = []
        for step in range(steps):
            ids = torch.randint(len(train), (batch_size,), device=target_device)
            x0 = train_tensor[ids]
            t = torch.randint(training_timesteps, (batch_size,), device=target_device)
            noise = torch.randn_like(x0)
            loss = F.mse_loss(model(schedule.add_noise(x0, noise, t), t), noise)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite diffusion training loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 100 == 0 or step + 1 == steps:
                history.append({"step": step + 1, "noise_mse": float(loss.detach())})
                print(json.dumps(history[-1]), flush=True)
        final = score()
        metadata.validation = {
            "metric": "epsilon_mse_on_fixed_heldout_corruption",
            "initial": initial,
            "final": final,
            "improved": final < initial,
            "noise_seed": seed + 2,
            "training_history": history,
        }
        save_prior(output, model, metadata)
    return metadata.model_dump(mode="json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--shape", type=int, nargs=2, default=(16, 16))
    parser.add_argument("--seed", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--images", help="NPZ with train_speed_mps and validation_speed_mps"
    )
    args = vars(parser.parse_args())
    args["shape"] = tuple(args["shape"])
    print(json.dumps(train_prior(**args), indent=2), flush=True)


if __name__ == "__main__":
    main()
