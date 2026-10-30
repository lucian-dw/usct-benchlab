"""Deployment and evaluation settings, separate from scientific parameters."""

from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field


class Deployment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    runtime_root: str | None = None
    matlab_executable: str = "matlab"
    cuda_visible_devices: str | None = None
    backend: Literal["cpu", "gpu"] = "gpu"
    scratch_root: str | None = None
    green_device: int | None = Field(
        None,
        ge=0,
        strict=True,
        description="Host-selected CuPy GPU index for volume-integral Born; None uses CPU",
    )


class EvaluationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    receiver_indices: list[Annotated[int, Field(ge=0, strict=True)]] | None = None
    frequency_indices: list[Annotated[int, Field(ge=0, strict=True)]] | None = None
    receiver_fraction: float = Field(0, ge=0, lt=1)
    frequency_fraction: float = Field(0, ge=0, lt=1)
    seed: int = Field(0, ge=0, strict=True)
    exclude_reciprocal: bool = True


class ImageEvaluationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    primary_region: Literal["tissue", "full_image"] = "tissue"
    water_speed_mps: float = Field(1500, gt=0)
    water_tolerance_mps: float = Field(0.1, ge=0)
    data_range_mps: float | None = Field(None, gt=0)
