"""SIREN: sinusoidal-activation coordinate MLP (Sitzmann et al., 2020)."""

import math
import torch
import torch.nn as nn


class SineLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        is_first: bool = False,
        omega_0: float = 30.0,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            if is_first:
                bound = 1.0 / in_features
            else:
                bound = math.sqrt(6.0 / in_features) / omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    def __init__(
        self,
        in_features: int = 2,
        out_features: int = 1,
        hidden_features: int = 256,
        hidden_layers: int = 4,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
        final_activation: str = "sigmoid",
    ):
        super().__init__()
        layers = [
            SineLayer(
                in_features, hidden_features, is_first=True, omega_0=first_omega_0
            )
        ]
        for _ in range(hidden_layers - 1):
            layers.append(
                SineLayer(hidden_features, hidden_features, omega_0=hidden_omega_0)
            )
        final = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_features) / hidden_omega_0
            final.weight.uniform_(-bound, bound)
        layers.append(final)
        if final_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif final_activation == "tanh":
            layers.append(nn.Tanh())
        elif final_activation is not None and final_activation != "identity":
            raise ValueError(f"Unknown final_activation: {final_activation!r}")
        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        return self.net(coords)
