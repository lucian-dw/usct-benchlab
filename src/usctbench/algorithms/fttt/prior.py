"""Explicit, metadata-checked, weights-only diffusion checkpoints."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Literal

from usctbench._vendor.inv_framework.models.tiny_unet import TinyUNet
from usctbench._vendor.inv_framework.solvers.diffusion.scheduler import VPSchedule


class PriorMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    format: Literal["usct.fttt.prior.v1"] = "usct.fttt.prior.v1"
    architecture: Literal["tiny_unet"] = "tiny_unet"
    prediction_type: Literal["epsilon"] = "epsilon"
    normalization: Literal["slowness_unit_interval_v1"] = "slowness_unit_interval_v1"
    image_shape: tuple[int, int]
    sound_speed_bounds_mps: tuple[float, float]
    base_channels: int = Field(8, ge=8, multiple_of=8)
    time_embedding_dim: int = Field(32, ge=2, multiple_of=2)
    training_timesteps: int = Field(1000, ge=2)
    beta_start: float = Field(0.0001, gt=0, lt=1)
    beta_end: float = Field(0.02, gt=0, lt=1)
    trained_steps: int = Field(gt=0)
    training_data: dict
    validation: dict

    @model_validator(mode="after")
    def compatible(self):
        if any(n < 8 or n % 8 for n in self.image_shape):
            raise ValueError("TinyUNet image_shape must be >=8 and divisible by 8")
        lo, hi = self.sound_speed_bounds_mps
        if not 0 < lo < hi or self.beta_start > self.beta_end:
            raise ValueError("invalid prior bounds or schedule")
        if not self.training_data or not self.validation:
            raise ValueError(
                "prior requires training provenance and held-out validation"
            )
        return self


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_prior(checkpoint, shape, bounds, device):
    if not checkpoint:
        raise ValueError(
            "DPS requires a pretrained checkpoint; no random-weight fallback"
        )
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"DPS checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"metadata", "state_dict"}:
        raise ValueError("expected checkpoint with metadata and state_dict")
    metadata = PriorMetadata.model_validate(payload["metadata"], strict=True)
    if metadata.image_shape != tuple(shape):
        raise ValueError(
            "DPS checkpoint image_shape differs from case grid; no implicit resizing"
        )
    if not np.allclose(metadata.sound_speed_bounds_mps, bounds, rtol=1e-12, atol=0):
        raise ValueError("DPS checkpoint normalization bounds differ from request")
    model = TinyUNet(1, metadata.base_channels, metadata.time_embedding_dim)
    model.load_state_dict(payload["state_dict"], strict=True)
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("DPS checkpoint contains nonfinite weights")
    model.to(device).eval().requires_grad_(False)
    schedule = VPSchedule(
        metadata.training_timesteps,
        metadata.beta_start,
        metadata.beta_end,
        device=device,
    )
    if float(schedule.alphas_cumprod[-1]) > 1e-3:
        raise ValueError(
            "DPS pure-noise initialization requires a near-Gaussian terminal training distribution"
        )
    record = metadata.model_dump(mode="json")
    record.update(checkpoint=str(path), sha256=file_sha256(path))
    return model, schedule, record


def save_prior(path, model, metadata):
    metadata = PriorMetadata.model_validate(metadata)
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to replace checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata.model_dump(),
        "state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
    }
    with path.open("xb") as handle:
        torch.save(payload, handle)
    manifest = metadata.model_dump(mode="json")
    manifest["sha256"] = file_sha256(path)
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
