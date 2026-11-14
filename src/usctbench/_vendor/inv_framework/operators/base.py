"""Abstract forward operators for inverse problems y = A(x) + n.

Two layers of abstraction:

* `ForwardOperator` — the most general case. A may be non-linear (phase
  retrieval, blind deblurring, saturated sensing, MRI with non-Cartesian
  trajectories, etc.). The only requirement is that `forward` is
  autograd-traceable, so that gradient-based solvers (DIP, INR, DPS, RED,
  variational methods) can backprop through L = loss(A(x), y).

* `LinearOperator` — A is linear. Adds `adjoint` (transpose-vector product)
  and a default `pseudo_inverse`. Solvers that require an explicit adjoint
  (FBP, SIRT, Landweber, PiGDM, ...) take this subtype.

CT is a linear inverse problem, so the operators in `operators/ct/` inherit
from `LinearOperator`. Non-linear operators inherit from `ForwardOperator`
directly.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import torch


class ForwardOperator(ABC):
    """Abstract forward model A: x -> y, linear or non-linear.

    Attributes
    ----------
    domain_shape : tuple
        Per-sample shape of x (no batch dim), e.g. ``(C, H, W)``.
    range_shape : tuple
        Per-sample shape of y.
    """

    domain_shape: Tuple[int, ...]
    range_shape: Tuple[int, ...]

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply A to a batched x. Must support autograd through x."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class LinearOperator(ForwardOperator):
    """Linear forward operator with an explicit adjoint.

    Implementations must wire `forward` so that its autograd backward equals
    `adjoint` — this lets gradient-based solvers and explicit-adjoint solvers
    agree on what A^T is. For CT operators we achieve this with a custom
    `torch.autograd.Function`; see `operators/ct/radon_torch.py`.
    """

    @abstractmethod
    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply A^T to y. Must support batched input."""

    def pseudo_inverse(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """Default pseudo-inverse via Landweber iteration.

        Subclasses may override (e.g. CT operators expose FBP/SIRT).
        """
        from inv_framework.solvers.classical import landweber

        return landweber(self, y, **kwargs)

    def project(
        self, x: torch.Tensor, y: torch.Tensor, step_size: Optional[float] = None
    ) -> torch.Tensor:
        """One Landweber-style data-consistency step toward Ax = y."""
        residual = self.forward(x) - y
        if step_size is None:
            step_size = 1.0
        return x - step_size * self.adjoint(residual)
