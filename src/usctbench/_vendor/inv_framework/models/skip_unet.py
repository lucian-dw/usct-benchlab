"""Lightweight U-Net with skip connections for Deep Image Prior."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SkipUNet(nn.Module):
    """3-level U-Net for DIP. Output passed through sigmoid for [0, 1]."""

    def __init__(self, in_channels: int = 32, out_channels: int = 1, base: int = 64):
        super().__init__()
        self.enc1 = _ConvBlock(in_channels, base)
        self.enc2 = _ConvBlock(base, base * 2)
        self.enc3 = _ConvBlock(base * 2, base * 4)
        self.bottleneck = _ConvBlock(base * 4, base * 8)
        self.dec3 = _ConvBlock(base * 8 + base * 4, base * 4)
        self.dec2 = _ConvBlock(base * 4 + base * 2, base * 2)
        self.dec1 = _ConvBlock(base * 2 + base, base)
        self.head = nn.Conv2d(base, out_channels, 1)
        self.pool = nn.AvgPool2d(2)

    @staticmethod
    def _up(x):
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, z):
        e1 = self.enc1(z)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self._up(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self._up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self._up(d2), e1], dim=1))
        return torch.sigmoid(self.head(d1))
