"""Deep Image Prior solver.

Fit a CNN f_theta(z) (fixed random z) so that A f_theta(z) matches y. The
network's implicit smoothness prior regularises the otherwise underdetermined
inverse problem. Reference: Ulyanov et al., 2018.
"""

from typing import Callable, Optional
import torch
import torch.nn as nn

from ..operators.base import ForwardOperator
from ..models.skip_unet import SkipUNet
from .base import InverseProblemSolver


class DIPSolver(InverseProblemSolver):
    """Deep Image Prior.

    Parameters
    ----------
    model_factory : callable, optional
        Returns an nn.Module mapping z -> x. Defaults to SkipUNet.
    num_iterations : int
    lr : float
    input_channels : int
        Channels of the fixed random input z.
    log_every : int, optional
    callback : callable(it, x_est, loss), optional
    """

    def __init__(
        self,
        model_factory: Optional[Callable[[], nn.Module]] = None,
        num_iterations: int = 2000,
        lr: float = 1e-3,
        input_channels: int = 32,
        log_every: Optional[int] = None,
        callback: Optional[Callable] = None,
    ):
        self.model_factory = model_factory
        self.num_iterations = num_iterations
        self.lr = lr
        self.input_channels = input_channels
        self.log_every = log_every
        self.callback = callback

    def solve(
        self, measurement: torch.Tensor, operator: ForwardOperator, **kwargs
    ) -> torch.Tensor:
        device = measurement.device
        B = measurement.shape[0]
        C, H, W = operator.domain_shape

        if self.model_factory is None:
            model = SkipUNet(in_channels=self.input_channels, out_channels=C).to(device)
        else:
            model = self.model_factory().to(device)

        z = torch.randn(B, self.input_channels, H, W, device=device)
        optim = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = kwargs.get("loss_fn")
        observer = kwargs.get("observer")
        before_step = kwargs.get("before_step", lambda: None)
        if observer is not None:
            with torch.no_grad():
                if observer(0, model(z)):
                    return model(z)

        for it in range(self.num_iterations):
            before_step()
            optim.zero_grad()
            x_est = model(z)
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
                    if observer(it + 1, model(z)):
                        break

            if self.log_every and (
                it % self.log_every == 0 or it == self.num_iterations - 1
            ):
                print(
                    f"[DIP] iter {it}/{self.num_iterations} " f"loss={loss.item():.4e}"
                )
            if self.callback is not None:
                self.callback(it, x_est.detach(), loss.item())

        with torch.no_grad():
            return model(z)
