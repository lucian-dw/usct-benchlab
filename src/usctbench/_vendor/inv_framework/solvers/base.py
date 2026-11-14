"""Abstract solver for inverse problems."""

from abc import ABC, abstractmethod
import torch

from ..operators.base import ForwardOperator


class InverseProblemSolver(ABC):
    """Given a measurement y and operator A, return an estimate of x.

    All solvers in this framework expose the same `solve` signature so that
    classical, model-based (DIP, INR), and diffusion methods are
    interchangeable in benchmarks. Solvers that require linearity (e.g.
    FBP, SIRT, PiGDM) take a `LinearOperator` and assert that internally;
    everything else accepts any `ForwardOperator`.
    """

    @abstractmethod
    def solve(
        self, measurement: torch.Tensor, operator: ForwardOperator, **kwargs
    ) -> torch.Tensor: ...
