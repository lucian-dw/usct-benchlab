"""Authoritative native parameter models; constraints are numerical, not presets.

Null defaults denote case-dependent values and are omitted at the solver boundary.
Budgets and evaluation policies are validated separately in ``configuration``.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Positive = Annotated[float, Field(strict=True, gt=0)]
Nonnegative = Annotated[float, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonnegativeInt = Annotated[int, Field(strict=True, ge=0)]
Bounds = tuple[Positive, Positive]


def parameter(default, description, units="1", exposure="advanced", **kwargs):
    return Field(
        default=default,
        description=description,
        json_schema_extra={"units": units, "exposure": exposure},
        **kwargs,
    )


class Parameters(BaseModel):
    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False, validate_default=True
    )

    @model_validator(mode="after")
    def ordered_bounds(self):
        bounds = getattr(self, "sound_speed_bounds_mps", None)
        if bounds is not None and bounds[0] >= bounds[1]:
            raise ValueError("sound_speed_bounds_mps requires positive low < high")
        return self


class SoundSpeedParameters(Parameters):
    reference_sound_speed_mps: Positive | None = parameter(
        None,
        "Measurement reference; case metadata, otherwise 1500 m/s.",
        "m/s",
        "internal",
    )
    sound_speed_bounds_mps: Bounds = parameter(
        (1300.0, 1700.0),
        "Ordered numerical projection bounds, not a calibrated physiological range.",
        "m/s",
        "agent",
    )
    roi_update_only: bool = parameter(
        False, "Restrict updates to the supplied ROI, if present."
    )


class RayParameters(SoundSpeedParameters):
    model_grid_shape: tuple[PositiveInt, PositiveInt] | None = parameter(
        None,
        "Optional coefficient grid [row=y, col=x]; null uses the case grid.",
        "pixels",
    )
    min_ray_weight: Annotated[float, Field(strict=True, ge=0, le=1)] = parameter(
        0.0, "Reject ray confidence below this threshold."
    )
    ray_weight_power: Positive = parameter(
        1.0, "Exponent applied to confidence weights."
    )


class CGLSParameters(RayParameters):
    regularization: Literal["identity", "laplacian"] = parameter(
        "identity", "Penalty operator L in lambda^2 ||L x||^2.", exposure="agent"
    )
    regularization_lambda: Nonnegative = parameter(
        0.0,
        "Penalty amplitude in lambda^2 ||L x||^2.",
        "objective/operator dependent",
    )
    robust_loss: Literal["none", "huber"] = parameter(
        "none", "Quadratic residual or Huber IRLS.", exposure="agent"
    )
    huber_delta_s: Positive = parameter(
        5e-7, "Huber transition on travel-time residual.", "s"
    )
    irls_max_inner_steps: PositiveInt = parameter(
        10, "Maximum CG steps per weight refresh, only used with Huber.", "steps"
    )
    gradient_rtol: Nonnegative = parameter(
        1e-10, "Relative projected optimality tolerance."
    )
    roi_laplacian: bool = parameter(False, "Use ROI-aware Laplacian regularization.")
    coverage_preconditioning: bool = parameter(
        False, "Enable diagonal coverage preconditioning."
    )
    coverage_preconditioner_eps: Positive = parameter(
        1e-12, "Coverage denominator floor.", "coverage units"
    )
    coverage_preconditioner_max_scale: Positive = parameter(
        10.0, "Maximum diagonal preconditioner scale."
    )
    coverage_preconditioner_normalize: bool = parameter(
        True, "Normalize positive ROI coverage scales by their median."
    )


class SIRTParameters(RayParameters):
    relaxation: Annotated[float, Field(strict=True, gt=0, lt=2)] = parameter(
        0.3,
        "Algebraic update relaxation; stability also depends on smoothing/projection.",
        exposure="agent",
    )
    smooth_sigma: Nonnegative = parameter(
        0.0, "Gaussian smoothing standard deviation.", "pixels"
    )
    smooth_every: NonnegativeInt = parameter(
        0, "Smoothing interval; zero disables smoothing.", "complete iterations"
    )


class SARTParameters(SIRTParameters):
    relaxation: Annotated[float, Field(strict=True, gt=0, lt=2)] = parameter(
        0.2, "Subset update relaxation.", exposure="agent"
    )
    subsets: PositiveInt = parameter(
        8, "Number of ordered subsets in one complete sweep.", "subsets"
    )


class InnerOptions(Parameters):
    rtol: Nonnegative = parameter(1e-7, "Normal residual tolerance.")
    atol: Nonnegative = parameter(
        1e-10, "Augmented least-squares absolute solver tolerance."
    )
    btol: Nonnegative = parameter(
        1e-10, "Augmented least-squares relative solver tolerance."
    )
    conlim: Nonnegative = parameter(1e8, "LSMR/LSQR condition limit, zero disables.")
    preconditioner: Literal["none", "column_rms"] = parameter(
        "none", "Inner column preconditioner."
    )
    preconditioner_probes: PositiveInt = parameter(
        8, "Diagonal estimation probes.", "probes"
    )
    preconditioner_seed: NonnegativeInt = parameter(0, "Deterministic probe seed.")


class GaussNewtonParameters(SoundSpeedParameters):
    inner_iterations: PositiveInt = parameter(
        16,
        "Truncated inner linear solve budget, not a nonlinear iteration count.",
        "inner iterations",
    )
    inner_solver: Literal["lsmr", "lsqr", "normal_cg"] = parameter(
        "lsmr", "Inner least-squares method."
    )
    inner_options: InnerOptions = parameter(
        InnerOptions(), "Inner numerical controls; never exposed to default Agent."
    )
    step_length: Positive = parameter(1.0, "Initial line-search step length.")
    smooth_sigma: Nonnegative = parameter(
        0.0, "Update smoothing standard deviation.", "pixels"
    )
    gradient_rtol: Nonnegative = parameter(
        1e-8, "Relative training gradient tolerance."
    )


class BentParameters(GaussNewtonParameters, RayParameters):
    initialization: Literal["configured", "cgls"] = parameter(
        "configured",
        "Scalar initialization or counted CGLS initialization.",
        exposure="agent",
    )
    initial_sound_speed_mps: Positive | list[list[Positive]] | None = parameter(
        None, "Null uses reference speed; expert calls may supply a [y,x] image.", "m/s"
    )
    initialization_iterations: PositiveInt = parameter(
        80, "CGLS setup iterations charged to work budgets.", "iterations"
    )
    regularization: Literal["identity", "laplacian"] = parameter(
        "laplacian", "Penalty on slowness perturbation.", exposure="agent"
    )
    regularization_lambda: Nonnegative = parameter(
        0.02,
        "Penalty amplitude in lambda^2 ||L x||^2.",
        "objective/operator dependent",
    )
    line_search: bool = parameter(True, "Enable objective backtracking.")
    eikonal_order: Literal[1, 2] = parameter(1, "Fast-marching spatial order.")


class BornPhysicsParameters(SoundSpeedParameters):
    """Shared measured-pressure physics; variant is the only physical selector."""

    background_sound_speed_mps: Positive | list[list[Positive]] | None = parameter(
        None,
        "Case reference by default; expert calls may supply a [y,x] background.",
        "m/s",
    )

    regularization_lambda: Nonnegative = parameter(
        0.0,
        "Penalty amplitude before optional Jacobian scaling.",
        "objective/operator dependent",
    )

    regularization_scaling: Literal["absolute", "relative_jacobian_diagonal"] = (
        parameter("absolute", "Penalty scaling convention.")
    )

    regularization_length_wavelengths: Nonnegative = parameter(
        0.0,
        "Laplacian length in training-band wavelengths; zero uses legacy pixel penalty.",
        "wavelengths",
        "agent",
    )

    green_solver_rtol: Annotated[float, Field(strict=True, gt=0, lt=1)] = parameter(
        1e-7, "Volume-integral Green solver tolerance."
    )

    green_solver_maxiter: PositiveInt = parameter(
        20, "Volume-integral Green solver iteration cap.", "iterations"
    )

    allow_underresolved: bool = parameter(
        False, "Expert override of the inverse-grid PPW guard."
    )

    assume_unit_source: bool = parameter(
        False, "Only for deliberately unit-source synthetic data.", exposure="internal"
    )

    use_feature_weights: bool = parameter(
        False, "Use feature-derived pressure weights, subject to leakage checks."
    )

    max_cache_bytes: NonnegativeInt = parameter(
        128 * 1024**2, "Green cache capacity.", "bytes", "internal"
    )

    @property
    def green_backend(self):
        return (
            "volume_integral"
            if self.variant.startswith("full_green")
            else "eikonal_wkb"
        )

    @property
    def mode(self):
        return "fixed_background" if self.variant.endswith("_fixed") else "nonlinear"


class FixedBornParameters(BornPhysicsParameters):
    variant: Literal["wkb_fixed", "full_green_fixed"] = parameter(
        "wkb_fixed", "Fixed-background Green model.", exposure="agent"
    )
    regularization: Literal["identity", "laplacian"] = parameter(
        "identity", "Squared-slowness penalty operator.", exposure="agent"
    )


class BornParameters(BornPhysicsParameters, GaussNewtonParameters):
    variant: Literal["wkb_nonlinear", "full_green_nonlinear"] = parameter(
        "wkb_nonlinear", "Relinearized Green model.", exposure="agent"
    )
    regularization: Literal["identity", "laplacian"] = parameter(
        "laplacian", "Penalty on squared-slowness perturbation.", exposure="agent"
    )
    initialization: Literal["configured", "phase_cgls"] = parameter(
        "configured", "Background initialization method.", exposure="agent"
    )

    initialization_iterations: PositiveInt = parameter(
        80, "Counted phase-CGLS setup budget.", "iterations"
    )

    initialization_lambda: Nonnegative = parameter(
        0.02, "Initialization regularization amplitude.", "objective/operator dependent"
    )

    initialization_smooth_mm: Nonnegative = parameter(
        3.0, "Initialization smoothing scale.", "mm"
    )

    initialization_max_phase_rms: Positive = parameter(
        0.2, "Phase-fit acceptance RMS.", "rad"
    )

    initialization_min_amplitude_ratio: Nonnegative = parameter(
        0.05, "Minimum usable phase amplitude ratio."
    )

    inner_iterations: PositiveInt = parameter(
        12,
        "Truncated nonlinear inner linear-solve budget.",
        "inner iterations",
    )

    max_update_mps: Positive = parameter(
        12.0, "Bound on each nonlinear sound-speed update.", "m/s", "agent"
    )

    max_backtracks: NonnegativeInt = parameter(
        10, "Maximum backtracks after the initial trial; zero tries once.", "backtracks"
    )
