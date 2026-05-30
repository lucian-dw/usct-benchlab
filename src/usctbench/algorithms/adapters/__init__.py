"""Optional wrappers for established external USCT methods."""

from __future__ import annotations

from usctbench.registry import register_algorithm

from .refraction_gn import BentRayGNAdapter
from .rwave import RWaveAdapter


def register_adapter_algorithms(*, replace: bool = False) -> None:
    register_algorithm(
        "bent_ray_gn",
        BentRayGNAdapter,
        description="Optional MATLAB refraction-corrected Gauss-Newton adapter.",
        tags=("adapter", "matlab", "refraction"),
        replace=replace,
    )
    register_algorithm(
        "rwave_adapter",
        RWaveAdapter,
        description="Optional MATLAB ray-Born/r-Wave adapter.",
        tags=("adapter", "matlab", "ray-born"),
        replace=replace,
    )


__all__ = ["BentRayGNAdapter", "RWaveAdapter", "register_adapter_algorithms"]

