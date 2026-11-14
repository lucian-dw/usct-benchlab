"""Typed scientific controls for the extracted DIP, INR and DPS methods."""

from typing import Annotated, Literal
from pydantic import Field, model_validator

from usctbench.algorithms.parameters import (
    Nonnegative,
    NonnegativeInt,
    Positive,
    PositiveInt,
    SoundSpeedParameters,
    parameter,
)


class FTTTParameters(SoundSpeedParameters):
    min_ray_weight: Annotated[float, Field(strict=True, ge=0, le=1)] = parameter(
        0.0, "Reject ray confidence below this threshold."
    )
    ray_weight_power: Positive = parameter(
        1.0, "Exponent applied to confidence weights."
    )
    seed: NonnegativeInt = parameter(0, "Local PyTorch random seed.", exposure="agent")
    eikonal_order: Literal[1, 2] = parameter(1, "Fast-marching spatial order.")
    residual_scale_s: Positive = parameter(
        1e-6,
        "Fixed seconds per normalized residual; never estimated from holdouts.",
        "s",
        "agent",
    )


class DIPParameters(FTTTParameters):
    variant: Literal["dip"] = parameter("dip", "Deep Image Prior.", exposure="agent")
    learning_rate: Positive = parameter(0.001, "Adam learning rate.", exposure="agent")
    input_channels: PositiveInt = parameter(8, "Fixed random input channels.")
    base_channels: PositiveInt = parameter(16, "SkipUNet base channels.")


class INRParameters(FTTTParameters):
    variant: Literal["inr"] = parameter("inr", "Coordinate SIREN.", exposure="agent")
    learning_rate: Positive = parameter(0.0001, "Adam learning rate.", exposure="agent")
    hidden_features: PositiveInt = parameter(64, "SIREN hidden layer width.")
    hidden_layers: PositiveInt = parameter(3, "SIREN hidden layer count.")


class DPSParameters(FTTTParameters):
    variant: Literal["dps"] = parameter(
        "dps", "Diffusion Posterior Sampling.", exposure="agent"
    )
    sampling_steps: PositiveInt = parameter(
        64,
        "Full reverse schedule length; host iteration caps truncate this schedule.",
        "sampling steps",
        "agent",
    )
    guidance_scale: Nonnegative = parameter(
        0.1,
        "Step scale for the normalized weighted residual-norm gradient.",
        exposure="agent",
    )
    eta: Nonnegative = parameter(0.0, "DDIM stochasticity, in [0,1].")
    checkpoint: str | None = parameter(
        None,
        "Host-provided usct.fttt.prior.v1 checkpoint; select this or prior_manifest.",
        exposure="internal",
    )
    prior_manifest: str | None = parameter(
        None,
        "Host-selected dataset manifest in diffusion/weights/<dataset>/manifest.json.",
        exposure="internal",
    )

    @model_validator(mode="after")
    def valid_eta(self):
        if self.eta > 1:
            raise ValueError("eta must be in [0,1]")
        if self.checkpoint and self.prior_manifest:
            raise ValueError("select checkpoint or prior_manifest, not both")
        return self

    def validate_run_controls(self, controls):
        if controls.objective_rtol is not None:
            raise ValueError("DPS does not support objective-plateau stopping")
