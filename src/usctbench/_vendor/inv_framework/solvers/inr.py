"""Implicit Neural Representation (INR) solver.

Parametrise x as a coordinate-MLP f_theta(coord). Optimise theta so that
A f_theta matches y. Defaults to SIREN.
"""

from typing import Callable, Optional
import torch
import torch.nn as nn

from ..operators.base import ForwardOperator
from ..models.siren import SIREN
from .base import InverseProblemSolver


class INRSolver(InverseProblemSolver):
    def __init__(
        self,
        model_factory: Optional[Callable[[], nn.Module]] = None,
        num_iterations: int = 2000,
        lr: float = 1e-4,
        hidden_features: int = 256,
        hidden_layers: int = 4,
        log_every: Optional[int] = None,
    ):
        self.model_factory = model_factory
        self.num_iterations = num_iterations
        self.lr = lr
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.log_every = log_every

    def solve(
        self, measurement: torch.Tensor, operator: ForwardOperator, **kwargs
    ) -> torch.Tensor:
        device = measurement.device
        B = measurement.shape[0]
        C, H, W = operator.domain_shape

        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij",
        )
        coords = torch.stack([xs, ys], dim=-1).reshape(-1, 2)  # (H*W, 2)

        if self.model_factory is None:
            model = SIREN(
                in_features=2,
                out_features=C,
                hidden_features=self.hidden_features,
                hidden_layers=self.hidden_layers,
            ).to(device)
        else:
            model = self.model_factory().to(device)

        optim = torch.optim.Adam(model.parameters(), lr=self.lr)

        def render() -> torch.Tensor:
            out = model(coords).view(H, W, C).permute(2, 0, 1).unsqueeze(0)
            return out.expand(B, -1, -1, -1)

        loss_fn = kwargs.get("loss_fn")
        observer = kwargs.get("observer")
        before_step = kwargs.get("before_step", lambda: None)
        if observer is not None:
            with torch.no_grad():
                if observer(0, render()):
                    return render()
        for it in range(self.num_iterations):
            before_step()
            optim.zero_grad()
            x_est = render()
            y_est = operator.forward(x_est)
            loss = (
                loss_fn(y_est, measurement)
                if loss_fn is not None
                else ((y_est - measurement) ** 2).mean()
            )
            loss.backward()
            optim.step()
            if observer is not None:
                with torch.no_grad():
                    if observer(it + 1, render()):
                        break

            if self.log_every and (
                it % self.log_every == 0 or it == self.num_iterations - 1
            ):
                print(
                    f"[INR] iter {it}/{self.num_iterations} " f"loss={loss.item():.4e}"
                )

        with torch.no_grad():
            return render()
