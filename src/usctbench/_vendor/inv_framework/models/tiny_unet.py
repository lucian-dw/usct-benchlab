"""Compact time-conditioned UNet for diffusion demos.

Predicts the noise epsilon given (x_t, t). Kept small (a few M params) so the
DPS demo can train end-to-end in a few minutes on a single GPU.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(
    t: torch.Tensor, dim: int, max_period: int = 10000
) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device).float() / half
    )
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class _TimeBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, t_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_c), out_c)
        self.norm2 = nn.GroupNorm(min(8, out_c), out_c)
        self.t_proj = nn.Linear(t_dim, out_c)
        self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x, t_emb):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.t_proj(F.silu(t_emb))[:, :, None, None]
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class TinyUNet(nn.Module):
    """A 3-level time-conditioned UNet."""

    def __init__(self, channels: int = 1, base: int = 32, t_dim: int = 128):
        super().__init__()
        self.t_dim = t_dim
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim)
        )

        self.enc1 = _TimeBlock(channels, base, t_dim)
        self.enc2 = _TimeBlock(base, base * 2, t_dim)
        self.enc3 = _TimeBlock(base * 2, base * 4, t_dim)
        self.mid = _TimeBlock(base * 4, base * 4, t_dim)
        self.dec3 = _TimeBlock(base * 4 + base * 4, base * 2, t_dim)
        self.dec2 = _TimeBlock(base * 2 + base * 2, base, t_dim)
        self.dec1 = _TimeBlock(base + base, base, t_dim)
        self.head = nn.Conv2d(base, channels, 1)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        emb = self.t_mlp(timestep_embedding(t, self.t_dim))
        e1 = self.enc1(x, emb)
        e2 = self.enc2(self.pool(e1), emb)
        e3 = self.enc3(self.pool(e2), emb)
        m = self.mid(self.pool(e3), emb)
        u3 = F.interpolate(m, scale_factor=2, mode="nearest")
        d3 = self.dec3(torch.cat([u3, e3], dim=1), emb)
        u2 = F.interpolate(d3, scale_factor=2, mode="nearest")
        d2 = self.dec2(torch.cat([u2, e2], dim=1), emb)
        u1 = F.interpolate(d2, scale_factor=2, mode="nearest")
        d1 = self.dec1(torch.cat([u1, e1], dim=1), emb)
        return self.head(d1)
