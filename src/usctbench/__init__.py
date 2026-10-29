"""USCT benchmark package."""

from __future__ import annotations

from .core.schema import (
    AlgorithmConfig,
    GeometrySpec,
    GridSpec,
    GroundTruthSpec,
    MeasurementSpec,
    ReconstructionResult,
    USCTCase,
)

__all__ = [
    "AlgorithmConfig",
    "GeometrySpec",
    "GridSpec",
    "GroundTruthSpec",
    "MeasurementSpec",
    "ReconstructionResult",
    "USCTCase",
]

from .core.registry import get_algorithm, get_algorithm_entry, list_algorithms
from .core.config import prepare_config
from .core.execution import Deployment
from .core.run_controls import RunControls
from .benchmark import run, suite, evaluate

__all__ += ["get_algorithm", "get_algorithm_entry", "list_algorithms"]


def describe(algorithm, variant=None):
    """Read the static model, parameters, observations and budget capabilities."""
    return get_algorithm_entry(algorithm).describe(variant)


__all__ += [
    "prepare_config",
    "Deployment",
    "RunControls",
    "run",
    "suite",
    "evaluate",
    "describe",
]
