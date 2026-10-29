"""Static catalog of maintained algorithms and their physical capabilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .schema import AlgorithmConfig, ReconstructionResult, USCTCase
from .algorithm_specs import AlgorithmSpecification, SPECS
from usctbench.algorithms.bent_ray import BentRayGNAdapter
from usctbench.algorithms.fwi import WUSTFWIAlgorithm
from usctbench.algorithms.ray import (
    StraightRayCGLSAlgorithm,
    StraightRaySIRTAlgorithm,
    StraightRaySARTAlgorithm,
)
from usctbench.algorithms.rwave import RWaveAdapter


class Algorithm(Protocol):
    name: str

    def run(self, case: USCTCase, config: AlgorithmConfig) -> ReconstructionResult:
        """Run one reconstruction from prepared observations and configuration."""


@dataclass(frozen=True)
class AlgorithmEntry:
    name: str
    factory: Callable[[], Algorithm]
    specification: AlgorithmSpecification

    @property
    def description(self):
        return self.specification.description

    @property
    def tags(self):
        return (self.specification.family,)

    def describe(self, variant: str | None = None) -> dict:
        return self.specification.describe(variant)


_REGISTRY = {
    name: AlgorithmEntry(name, factory, SPECS[name])
    for name, factory in (
        ("straight_cgls", StraightRayCGLSAlgorithm),
        ("straight_sirt", StraightRaySIRTAlgorithm),
        ("straight_sart", StraightRaySARTAlgorithm),
        ("bent_ray_gn", BentRayGNAdapter),
        ("rwave_adapter", RWaveAdapter),
        ("fwi_wust", WUSTFWIAlgorithm),
    )
}


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
