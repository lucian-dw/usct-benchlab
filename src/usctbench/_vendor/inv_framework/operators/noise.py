"""Measurement noise models."""

from abc import abstractmethod
import torch
import torch.nn as nn


class NoiseModel(nn.Module):
    """Abstract noise model y_noisy = noiser(y_clean)."""

    @abstractmethod
    def forward(self, y_clean: torch.Tensor) -> torch.Tensor: ...


class NoNoise(NoiseModel):
    def forward(self, y_clean):
        return y_clean


class GaussianNoise(NoiseModel):
    def __init__(self, sigma: float):
        super().__init__()
        self.sigma = sigma

    def forward(self, y_clean):
        return y_clean + self.sigma * torch.randn_like(y_clean)


class PoissonLogDomainNoise(NoiseModel):
    """Poisson noise in the log-attenuation (sinogram) domain.

    Models I = Poisson(I0 * exp(-mu * x)) and returns -log(I / I0), so the
    noisy measurement is still in the same domain as A(x). Adapted from DM4CT.
    """

    def __init__(self, photon_count: float, transmittance_target: float = None):
        super().__init__()
        self.photon_count = photon_count
        self.transmittance_target = transmittance_target

    def forward(self, sinogram):
        scale = 1.0
        if self.transmittance_target is not None:
            mean = sinogram.mean()
            if mean.item() > 0:
                target = torch.tensor(self.transmittance_target, device=sinogram.device)
                scale = (-torch.log(target) / mean).item()
        scaled = sinogram * scale
        intensity = torch.exp(-scaled)
        counts = torch.poisson(intensity * self.photon_count)
        counts = counts.clamp(min=1.0)
        log_atten = -torch.log(counts / self.photon_count)
        return log_atten / scale
