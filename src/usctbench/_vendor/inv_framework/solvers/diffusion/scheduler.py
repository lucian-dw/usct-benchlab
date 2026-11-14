"""Noise schedules for diffusion-based solvers.

The API mirrors HuggingFace `diffusers` so that schedulers from either side
can be swapped freely:

    sched.set_timesteps(N)
    for t in sched.timesteps:
        ...
        x_prev, x0_pred = sched.step(eps, t, x_t, eta=...)

`VPSchedule` provides a self-contained DDPM-linear-beta implementation with
no external dependency. To use a `diffusers` scheduler instead, wrap it with
`DiffusersScheduleAdapter` (see `diffusers_compat.py`).
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union
import torch


class NoiseSchedule(ABC):
    """Abstract noise schedule.

    Attributes
    ----------
    num_train_timesteps : int
    timesteps : torch.Tensor
        Reverse-process timesteps (descending). Populated by `set_timesteps`.
    """

    num_train_timesteps: int
    timesteps: torch.Tensor

    @abstractmethod
    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Configure the reverse schedule; sets `self.timesteps`."""

    @abstractmethod
    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t) -> torch.Tensor:
        """Sample x_t = sqrt(alpha_bar_t) x0 + sqrt(1 - alpha_bar_t) noise."""

    @abstractmethod
    def step(
        self, eps: torch.Tensor, t, x_t: torch.Tensor, eta: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One reverse step. Returns (x_prev, x_0_pred). Must preserve the
        autograd graph from `x_t` and `eps` so DPS-style guidance works."""

    def predict_x0_from_eps(
        self, x_t: torch.Tensor, eps: torch.Tensor, t
    ) -> torch.Tensor:
        """Tweedie: x_0_hat = (x_t - sqrt(1-a) eps) / sqrt(a)."""
        a = self._alpha_bar_view(t).to(x_t.device)
        return (x_t - (1.0 - a).sqrt() * eps) / a.sqrt()

    def _alpha_bar_view(self, t) -> torch.Tensor:
        raise NotImplementedError


class VPSchedule(NoiseSchedule):
    """Variance-preserving (DDPM/DDIM) schedule with linear beta."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: Union[str, torch.device] = "cpu",
    ):
        self.num_train_timesteps = int(num_train_timesteps)
        if self.num_train_timesteps < 2 or not 0 < beta_start <= beta_end < 1:
            raise ValueError(
                "schedule requires >=2 timesteps and 0 < beta_start <= beta_end < 1"
            )
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps, device=device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.device = device
        self.timesteps = torch.empty(0, dtype=torch.long, device=device)
        self._prev_t = {}

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        device = device if device is not None else self.device
        if not 1 <= num_inference_steps <= self.num_train_timesteps:
            raise ValueError("inference steps must be between 1 and training timesteps")
        all_t = torch.linspace(
            self.num_train_timesteps - 1, 0, num_inference_steps, device=device
        ).long()
        self.timesteps = all_t
        self._prev_t = {
            int(all_t[i].item()): (
                int(all_t[i + 1].item()) if i + 1 < num_inference_steps else -1
            )
            for i in range(num_inference_steps)
        }

    def _alpha_bar_view(self, t) -> torch.Tensor:
        if torch.is_tensor(t):
            a = self.alphas_cumprod[t]
        else:
            a = self.alphas_cumprod[int(t)]
        if a.ndim == 0:
            a = a.view(1)
        return a.view(-1, 1, 1, 1)

    def add_noise(self, x0, noise, t):
        a = self._alpha_bar_view(t).to(x0.device)
        return a.sqrt() * x0 + (1.0 - a).sqrt() * noise

    def step(self, eps, t, x_t, eta: float = 0.0):
        t_int = int(t.item()) if torch.is_tensor(t) else int(t)
        t_prev = self._prev_t.get(t_int, -1)

        a_t = self._alpha_bar_view(t_int).to(x_t.device)
        if t_prev < 0:
            a_prev = torch.ones_like(a_t)
        else:
            a_prev = self._alpha_bar_view(t_prev).to(x_t.device)

        x0 = (x_t - (1.0 - a_t).sqrt() * eps) / a_t.sqrt()
        sigma = (
            eta
            * ((1.0 - a_prev) / (1.0 - a_t).clamp(min=1e-12)).sqrt()
            * (1.0 - a_t / a_prev.clamp(min=1e-12)).clamp(min=0).sqrt()
        )
        dir_xt = (1.0 - a_prev - sigma**2).clamp(min=0).sqrt() * eps
        noise = torch.randn_like(x_t) if eta > 0 else 0.0
        x_prev = a_prev.sqrt() * x0 + dir_xt + sigma * noise
        return x_prev, x0
