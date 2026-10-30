"""Static catalog of maintained algorithms and their physical capabilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from .schema import AlgorithmConfig, ReconstructionResult, USCTCase
from .algorithm_specs import AlgorithmSpecification, SPECS, agent_schema
from usctbench.algorithms.bent_ray import BentRayAlgorithm
from usctbench.algorithms.fwi import WUSTFWIAlgorithm
from usctbench.algorithms.straight import (
    StraightRayCGLSAlgorithm,
    StraightRaySIRTAlgorithm,
    StraightRaySARTAlgorithm,
)
from usctbench.algorithms.born import BornAlgorithm
from usctbench.algorithms.parameters import (
    CGLSParameters,
    SIRTParameters,
    SARTParameters,
    BentParameters,
    BornParameters,
    FixedBornParameters,
)
from usctbench.algorithms.fwi.parameters import WUSTParameters
from usctbench.core.run_controls import BUDGET_FIELDS, RunControls


class Algorithm(Protocol):
    name: str

    def run(self, case: USCTCase, config: AlgorithmConfig) -> ReconstructionResult:
        """Run one reconstruction from prepared observations and configuration."""


@dataclass(frozen=True)
class AlgorithmEntry:
    name: str
    factory: Callable[[], Algorithm]
    specification: AlgorithmSpecification
    models: dict
    iteration_defaults: dict
    supported_budgets: tuple = BUDGET_FIELDS
    numerical_stopping: bool = True
    deadline_semantics: str = (
        "cooperative; launching process must enforce a hard deadline"
    )
    budget_descriptions: dict[str, str] = field(default_factory=dict)

    @property
    def variants(self):
        return self.specification.variants

    def parameter_model(self, parameters=None):
        selected = (
            parameters.get("variant")
            if isinstance(parameters, dict)
            else getattr(parameters, "variant", None)
        )
        variant = self.specification.variant(selected)
        return self.models[variant.id]

    def default_iterations(self, parameters=None, case=None):
        selected = (
            parameters.get("variant")
            if isinstance(parameters, dict)
            else getattr(parameters, "variant", None)
        )
        rule = self.iteration_defaults[self.specification.variant(selected).id]
        if rule != "frequency_schedule":
            return rule
        schedule = (
            parameters.get("frequency_schedule_hz")
            if isinstance(parameters, dict)
            else getattr(parameters, "frequency_schedule_hz", None)
        )
        if schedule is not None:
            return len(schedule)
        if case is None:
            return None
        frequencies = case.measurement.frequencies_hz
        return 0 if frequencies is None else len(frequencies)

    def validate_controls(self, controls):
        unsupported = {
            key
            for key in BUDGET_FIELDS
            if key not in self.supported_budgets and getattr(controls, key) is not None
        }
        if not self.numerical_stopping:
            unsupported.update(
                controls.model_dump(exclude_defaults=True).keys() - set(BUDGET_FIELDS)
            )
        if unsupported:
            raise ValueError(
                f"{self.name} does not support run controls: {sorted(unsupported)}"
            )

    def budget_schema(self):
        schema = RunControls.model_json_schema()
        schema["properties"] = {
            key: value
            for key, value in schema["properties"].items()
            if key in self.supported_budgets
        }
        for key, description in self.budget_descriptions.items():
            schema["properties"][key]["description"] = description
        return schema

    @property
    def description(self):
        return self.specification.description

    @property
    def tags(self):
        return (self.specification.family,)

    def describe(self, variant: str | None = None) -> dict:
        spec = self.specification
        selected = spec.variant(variant)
        model = self.models[selected.id]
        schema, defaults = agent_schema(
            model, selected.selectors, enabled=selected.agent_parameters
        )
        return {
            "schema_version": "usct.algorithm.v1",
            "algorithm_id": self.name,
            "family": spec.family,
            "variant": selected.id,
            "variants": [
                {"id": item.id, "model": item.model} for item in self.variants
            ],
            "description": self.description,
            "mathematical_model": selected.model,
            "required_observation_domains": list(selected.observation_domains),
            "required_observations": {
                "all_of": list(selected.required_observations),
                "any_of": [list(group) for group in selected.observation_alternatives],
            },
            "supported_geometries": list(spec.supported_geometries),
            "parameter_model": model.__name__,
            "config_schema": schema,
            "allowed_parameters": sorted(schema["properties"]),
            "default_parameters": defaults,
            "compute_budget": {
                "separate_from_parameters": True,
                "default_max_iterations": self.default_iterations(selected.selectors),
                "online_controls_supported": bool(self.supported_budgets),
                "numerical_convergence_stopping_supported": self.numerical_stopping,
                "run_controls_schema": self.budget_schema(),
                "host_limits_schema": self.budget_schema(),
                "deadline_semantics": self.deadline_semantics,
            },
            "runtime_requirements": list(selected.runtime_requirements),
            "runtime_availability": "not_checked_static_description",
            "limitations": [
                "Research-grade within tested models/ranges; not clinical validation.",
                "Geometry acceptance does not establish imaging performance for every layout.",
                *selected.limitations,
            ],
            "iteration_unit": selected.iteration_unit,
            "deterministic_status": selected.deterministic,
        }


_REGISTRY = {}
for name, factory, model, default in (
    ("straight_cgls", StraightRayCGLSAlgorithm, CGLSParameters, 30),
    ("straight_sirt", StraightRaySIRTAlgorithm, SIRTParameters, 50),
    ("straight_sart", StraightRaySARTAlgorithm, SARTParameters, 10),
    ("bent_ray_gn", BentRayAlgorithm, BentParameters, 4),
):
    spec = SPECS[name]
    models = {v.id: model for v in spec.variants}
    defaults = {v.id: default for v in spec.variants}
    _REGISTRY[name] = AlgorithmEntry(name, factory, spec, models, defaults)

_REGISTRY["born"] = AlgorithmEntry(
    "born",
    BornAlgorithm,
    SPECS["born"],
    {
        "wkb_nonlinear": BornParameters,
        "full_green_nonlinear": BornParameters,
        "wkb_fixed": FixedBornParameters,
        "full_green_fixed": FixedBornParameters,
    },
    {
        "wkb_nonlinear": 4,
        "full_green_nonlinear": 4,
        "wkb_fixed": 30,
        "full_green_fixed": 30,
    },
)
_REGISTRY["fwi_wust"] = AlgorithmEntry(
    "fwi_wust",
    WUSTFWIAlgorithm,
    SPECS["fwi_wust"],
    {"wust": WUSTParameters},
    {"wust": "frequency_schedule"},
    supported_budgets=("max_iterations", "max_elapsed_s"),
    numerical_stopping=False,
    deadline_semantics=(
        "hard global deadline including admission, ingestion, reconstruction and parsing"
    ),
    budget_descriptions={
        "max_iterations": (
            "Truncate the resolved frequency schedule; null runs the full schedule, "
            "zero returns validated initialization."
        ),
        "max_elapsed_s": (
            "One hard global deadline in seconds; an explicit request or deployment "
            "timeout cap is required."
        ),
    },
)


def get_algorithm_entry(name: str) -> AlgorithmEntry:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown algorithm {name!r}; available: {', '.join(sorted(_REGISTRY))}"
        ) from exc


def get_algorithm(name: str) -> Algorithm:
    return get_algorithm_entry(name).factory()


def list_algorithms() -> list[AlgorithmEntry]:
    return [_REGISTRY[name] for name in sorted(_REGISTRY)]
