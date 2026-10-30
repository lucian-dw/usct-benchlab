"""Static catalog of maintained algorithms and their physical capabilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .schema import AlgorithmConfig, ReconstructionResult, USCTCase
from .algorithm_specs import AlgorithmSpecification, SPECS
from usctbench.algorithms.bent_ray import BentRayAlgorithm
from usctbench.algorithms.fwi import WUSTFWIAlgorithm
from usctbench.algorithms.straight import (
    StraightRayCGLSAlgorithm,
    StraightRaySIRTAlgorithm,
    StraightRaySARTAlgorithm,
)
from usctbench.algorithms.rwave import RayBornAlgorithm
from usctbench.algorithms.parameters import (
    CGLSParameters,
    SIRTParameters,
    SARTParameters,
    BentParameters,
    BornParameters,
    FixedBornParameters,
)
from usctbench.algorithms.fwi.parameters import WUSTParameters
from usctbench.core.run_controls import BUDGET_FIELDS


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
        if not self.numerical_stopping:
            controls.validate_for_wust()

    @property
    def description(self):
        return self.specification.description

    @property
    def tags(self):
        return (self.specification.family,)

    def describe(self, variant: str | None = None) -> dict:
        return self.specification.describe(variant)


_REGISTRY = {}
for name, factory, model, default in (
    ("straight_cgls", StraightRayCGLSAlgorithm, CGLSParameters, 30),
    ("straight_sirt", StraightRaySIRTAlgorithm, SIRTParameters, 50),
    ("straight_sart", StraightRaySARTAlgorithm, SARTParameters, 10),
    ("bent_ray_gn", BentRayAlgorithm, BentParameters, 4),
    ("rwave_adapter", RayBornAlgorithm, BornParameters, 4),
    ("fwi_wust", WUSTFWIAlgorithm, WUSTParameters, "frequency_schedule"),
):
    spec = SPECS[name]
    models = {v.id: model for v in spec.variants}
    defaults = {v.id: default for v in spec.variants}
    if name == "rwave_adapter":
        for variant in ("wkb_fixed", "full_green_fixed"):
            models[variant] = FixedBornParameters
            defaults[variant] = 30
    external = name == "fwi_wust"
    _REGISTRY[name] = AlgorithmEntry(
        name,
        factory,
        spec,
        models,
        defaults,
        ("max_iterations", "max_elapsed_s") if external else BUDGET_FIELDS,
        numerical_stopping=not external,
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
