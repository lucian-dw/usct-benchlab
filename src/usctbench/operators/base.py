"""Small array-based contracts shared by inverse solvers and physics models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from usctbench.core.schema import GridSpec


class LinearOperator(Protocol):
    """Image-to-data linear map with its Euclidean discrete adjoint.

    Real parameters and complex measurements use Re(<J x, y>) = <x, J* y>.
    Nonlinear forward models expose a linearization, not an 'inverse adjoint'.
    """

    grid: GridSpec
    data_dtype: np.dtype

    def forward(self, image: np.ndarray) -> np.ndarray: ...

    def adjoint(self, data: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class Linearization:
    value: np.ndarray
    jacobian: LinearOperator
    derivative_kind: str = "exact_discrete"


@dataclass(frozen=True)
class LinearSolveInfo:
    method: str
    iterations: int
    stop_reason: str
    relative_residual: float
    residual_norm: float
    converged: bool
    scipy_stop_code: int | None = None
    condition_estimate: float | None = None
    normal_residual_norm: float | None = None
    normal_residual_scope: str | None = None
