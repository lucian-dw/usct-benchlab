"""Static, versioned research algorithm descriptions and Agent admission.

Registry consumers use explicit physical variants, never naming heuristics.
Static schemas contain no case data or deployment-specific paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json

from pydantic import ConfigDict, create_model

from usctbench.core.schema import AlgorithmConfig


@dataclass(frozen=True)
class Variant:
    id: str
    model: str
    selectors: dict = field(default_factory=dict)
    observation_domains: tuple[str, ...] = ("features",)
    required_observations: tuple[str, ...] = ("delta_tof_s",)
    observation_alternatives: tuple[tuple[str, ...], ...] = ()
    runtime_requirements: tuple[str, ...] = ("numpy", "scipy")
    limitations: tuple[str, ...] = ()
    iteration_unit: str = "iteration"
    deterministic: str = (
        "deterministic_for_fixed_input_config_and_numerical_environment"
    )
    agent_parameters: bool = True


@dataclass(frozen=True)
class AlgorithmSpecification:
    algorithm_id: str
    family: str
    description: str
    variants: tuple[Variant, ...]
    supported_geometries: tuple[str, ...] = ("ring", "linear", "custom")

    def variant(self, variant_id=None):
        if variant_id is None:
            return self.variants[0]
        for variant in self.variants:
            if variant.id == variant_id:
                return variant
        raise ValueError(f"unknown variant {variant_id!r} for {self.algorithm_id}")


def agent_schema(model, selectors=None, *, enabled=True):
    """Build the schema from exposed fields only, including only their definitions."""
    fields = {
        name: (info.annotation, info)
        for name, info in model.model_fields.items()
        if enabled and (info.json_schema_extra or {}).get("exposure") == "agent"
    }
    schema = create_model(
        f"{model.__name__}Agent",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    ).model_json_schema()
    for key, value in schema["properties"].items():
        value.update(getattr(model, "agent_schema_overrides", {}).get(key, {}))
        if key in (selectors or {}):
            value["const"] = selectors[key]
            value["default"] = selectors[key]
    return schema, {
        key: value["default"]
        for key, value in schema["properties"].items()
        if "default" in value
    }


def _born_variant(identifier, mode, backend):
    return Variant(
        identifier,
        (
            "distorted_born_volume_integral"
            if backend == "volume_integral"
            else "eikonal_wkb_ray_born"
        ),
        selectors={"variant": identifier},
        observation_domains=("frequency",),
        required_observations=("freq_data", "frequencies_hz"),
        observation_alternatives=(("source_spectrum",), ("water_reference",)),
        iteration_unit=(
            "Ray-Born outer step" if mode == "nonlinear" else "Ray-Born CGLS step"
        ),
        limitations=(
            "Not a reproduction of the upstream r-Wave MATLAB package.",
            "Requires calibrated complex pressure, Fourier convention and adequate PPW.",
            (
                "WKB sensitivity approximates continuous physics, not the exact discrete WKB derivative."
                if backend == "eikonal_wkb"
                else "Full-Green accuracy depends on the volume-integral solve tolerance and grid."
            ),
            (
                "Fixed-background Born linearization only."
                if mode == "fixed_background"
                else "Background Green fields are updated during inversion."
            ),
        ),
    )


SPECS = {
    "straight_cgls": AlgorithmSpecification(
        "straight_cgls",
        "straight_ray",
        "Bounded weighted straight-ray least squares with optional regularization/IRLS.",
        (
            Variant(
                "cgls",
                "A delta_slowness = delta_tof",
                iteration_unit="cgls_step",
                limitations=("Assumes straight travel paths; not a waveform model.",),
            ),
        ),
    ),
    "straight_sirt": AlgorithmSpecification(
        "straight_sirt",
        "straight_ray",
        "Simultaneous algebraic travel-time reconstruction.",
        (
            Variant(
                "sirt",
                "A delta_slowness = delta_tof",
                iteration_unit="full_sweep",
                limitations=(
                    "Smoothing is an iterative filter, not an explicit lambda penalty.",
                ),
            ),
        ),
    ),
    "straight_sart": AlgorithmSpecification(
        "straight_sart",
        "straight_ray",
        "Ordered-subset algebraic travel-time reconstruction.",
        (
            Variant(
                "sart",
                "A delta_slowness = delta_tof",
                iteration_unit="full_sweep",
                limitations=(
                    "One iteration includes all subsets; do not compare with one subset update.",
                ),
            ),
        ),
    ),
    "bent_ray_gn": AlgorithmSpecification(
        "bent_ray_gn",
        "eikonal",
        "Native fast-marching travel-time Gauss-Newton reconstruction.",
        (
            Variant(
                "eikonal_gn",
                "|grad T| = slowness",
                required_observations=(),
                observation_alternatives=(("delta_tof_s",), ("tof_s",)),
                iteration_unit="Gauss-Newton outer step",
                limitations=(
                    "First-arrival geometrical acoustics; no pressure/diffraction modeling.",
                    "Eikonal differentiation is local to the active numerical stencil.",
                ),
            ),
        ),
    ),
    "born": AlgorithmSpecification(
        "born",
        "ray_born",
        "Native complex-pressure fixed and nonlinear WKB or Full-Green Born variants.",
        tuple(
            _born_variant(identifier, mode, backend)
            for identifier, mode, backend in (
                ("wkb_nonlinear", "nonlinear", "eikonal_wkb"),
                ("full_green_nonlinear", "nonlinear", "volume_integral"),
                ("wkb_fixed", "fixed_background", "eikonal_wkb"),
                ("full_green_fixed", "fixed_background", "volume_integral"),
            )
        ),
    ),
    "fwi_wust": AlgorithmSpecification(
        "fwi_wust",
        "full_wave",
        "2-D frequency-domain sound-speed FWI through the WUST v2 runtime; no attenuation reconstruction.",
        (
            Variant(
                "wust",
                "source_projected_Helmholtz_frequency_continuation_NCG",
                observation_domains=("frequency",),
                required_observations=(
                    "freq_data",
                    "frequencies_hz",
                    "valid_mask",
                    "declared_pressure_convention",
                ),
                runtime_requirements=(
                    "MATLAB",
                    "WUST_v2_MATLAB_explicit_CPU_or_CUDA_Block_LU",
                ),
                limitations=(
                    "GPU is production; CPU is reference/debug only. No CPU fallback.",
                    "Schedule completion is not convergence; no final-model residual evaluation.",
                    "Only iteration and global elapsed-time budgets are supported.",
                    "Scalar/reference initialization available to Agent; map is expert-only.",
                ),
                iteration_unit="frequency_schedule_update",
                deterministic="fixed_runtime_and_linear_solver_environment",
            ),
        ),
    ),
}


def make_agent_config(
    algorithm_id,
    parameters=None,
    *,
    variant=None,
    run_controls=None,
    trusted_parameters=None,
):
    """Only declared Agent knobs plus independent host-supplied context."""
    from usctbench.core.registry import get_algorithm_entry

    entry = get_algorithm_entry(algorithm_id)
    selected = entry.specification.variant(variant)
    description = entry.describe(selected.id)
    supplied = dict(parameters or {})
    forbidden = supplied.keys() - set(description["allowed_parameters"])
    if forbidden:
        raise ValueError(f"parameters not exposed to Agent: {sorted(forbidden)}")
    for key, value in supplied.items():
        choices = description["config_schema"]["properties"][key].get("enum")
        if choices is not None and value not in choices:
            raise ValueError(f"parameter {key} must be one of {choices}")
    values = {**(trusted_parameters or {}), **supplied}
    for key, value in selected.selectors.items():
        if key in values and values[key] != value:
            raise ValueError(f"parameter {key} conflicts with selected variant")
        values[key] = value
    controls = dict(run_controls or {})
    if controls.keys() - set(entry.supported_budgets):
        raise ValueError("unsupported Agent run control")
    model = entry.parameter_model(values)
    typed = model.model_validate_json(json.dumps(values, allow_nan=False), strict=True)
    return AlgorithmConfig(name=algorithm_id, parameters=typed, run_controls=controls)


def case_capabilities(
    algorithm_id,
    case,
    *,
    variant=None,
    runtime_available=None,
):
    """Case facts are separate from static specifications and clinical validity."""
    selected = SPECS[algorithm_id].variant(variant)
    frequencies = case.measurement.frequencies_hz
    return {
        "schema_version": "usct.case_capabilities.v1",
        "algorithm_id": algorithm_id,
        "variant": selected.id,
        "case_id": case.case_id,
        "available_frequencies_hz": [] if frequencies is None else frequencies.tolist(),
        "calibration_availability": {
            "source_spectrum_present": case.measurement.source_spectrum is not None,
            "water_reference_present": case.measurement.water_reference is not None,
            "meaning": "presence only; operator-specific units/conventions still apply",
        },
        "runtime_available": runtime_available,
        "runtime_availability_source": (
            "not_checked" if runtime_available is None else "deployment_reported"
        ),
        "validation_is_independent_test": False,
    }
