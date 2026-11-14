"""Diffusion Posterior Sampling (DPS) solver.

Chung et al., 2023, https://arxiv.org/abs/2209.14687.
Requires a pretrained noise predictor eps_theta(x_t, t). Works with both the
built-in `VPSchedule` and any HuggingFace `diffusers` scheduler wrapped via
`DiffusersScheduleAdapter`.
"""

from typing import Optional
import torch
import torch.nn as nn

from ...operators.base import ForwardOperator
from ...operators.noise import NoiseModel
from ..base import InverseProblemSolver
from .scheduler import NoiseSchedule, VPSchedule
from .conditioning import ConditioningMethod, PosteriorSampling


class DPSSolver(InverseProblemSolver):
    """DPS reverse-sampling with measurement guidance.

    Parameters
    ----------
    model : nn.Module
        Pretrained noise predictor with signature ``eps = model(x_t, t)``.
        For HuggingFace UNets, wrap with `DiffusersUNetWrapper` to coerce the
        return type.
    schedule : NoiseSchedule, optional
        Defaults to a VPSchedule with 1000 timesteps. Diffusers schedulers
        may be passed via `DiffusersScheduleAdapter`.
    num_inference_steps : int
    scale : float
        DPS guidance scale.
    eta : float
        DDIM stochasticity (0 = deterministic).
    conditioning : ConditioningMethod, optional
        Defaults to PosteriorSampling (the original DPS gradient).
    """

    def __init__(
        self,
        model: nn.Module,
        schedule: Optional[NoiseSchedule] = None,
        num_inference_steps: int = 200,
        scale: float = 1.0,
        eta: float = 0.0,
        conditioning: Optional[ConditioningMethod] = None,
        log_every: Optional[int] = None,
    ):
        self.model = model
        self.schedule = schedule if schedule is not None else VPSchedule()
        self.num_inference_steps = int(num_inference_steps)
        self.scale = scale
        self.eta = eta
        self.conditioning = conditioning
        self.log_every = log_every

    def solve(
        self,
        measurement: torch.Tensor,
        operator: ForwardOperator,
        noiser: Optional[NoiseModel] = None,
        x_init: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        device = measurement.device
        B = measurement.shape[0]

        cond = (
            self.conditioning
            if self.conditioning is not None
            else PosteriorSampling(operator, noiser, scale=self.scale)
        )

        self.schedule.set_timesteps(self.num_inference_steps, device=device)

        if x_init is None:
            x_t = torch.randn((B, *operator.domain_shape), device=device)
        else:
            x_t = x_init.clone()

        self.model.eval()
        observer = kwargs.get("observer")
        before_step = kwargs.get("before_step", lambda: None)
        max_iterations = kwargs.get("max_iterations", self.num_inference_steps)
        if observer is not None:
            with torch.no_grad():
                if observer(0, x_t):
                    return x_t
        for i, t in enumerate(self.schedule.timesteps):
            if i >= max_iterations:
                break
            before_step()
            x_t = x_t.detach().requires_grad_(True)
            t_batch = (
                t.expand(B)
                if torch.is_tensor(t)
                else torch.full((B,), int(t), device=device, dtype=torch.long)
            )
            eps = self.model(x_t, t_batch)
            x_t_next, x_0_hat = self.schedule.step(eps, t, x_t, eta=self.eta)
            x_t_next, residual = cond.apply(x_t_next, x_0_hat, measurement, x_prev=x_t)
            x_t = x_t_next.detach()
            if observer is not None:
                with torch.no_grad():
                    if observer(i + 1, x_t):
                        break

            if self.log_every and (
                i % self.log_every == 0 or i == self.num_inference_steps - 1
            ):
                t_val = int(t.item()) if torch.is_tensor(t) else int(t)
                print(
                    f"[DPS] step {i}/{self.num_inference_steps} "
                    f"t={t_val} res={residual.item():.4e}"
                )

        return x_t
