"""EIT benchmark API, following the USCT discovery/run/evaluate conventions."""

from .core.config import AlgorithmConfig, Deployment, prepare_config
from .core.schema import EITCase, ReconstructionResult
from .core.io import load_case, save_case
from .core.registry import get_algorithm, get_algorithm_entry, list_algorithms
from .core.run_controls import RunControls
from .benchmark.runner import run, suite
from .evaluation.metrics import evaluate


def describe(algorithm="lbfgsb_tv", variant=None):
    return get_algorithm_entry(algorithm).describe(variant)


__all__ = [
    "AlgorithmConfig",
    "Deployment",
    "EITCase",
    "ReconstructionResult",
    "RunControls",
    "prepare_config",
    "load_case",
    "save_case",
    "get_algorithm",
    "get_algorithm_entry",
    "list_algorithms",
    "describe",
    "run",
    "suite",
    "evaluate",
]
