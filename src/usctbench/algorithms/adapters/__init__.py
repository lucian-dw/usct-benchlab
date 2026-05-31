"""Optional wrappers for established external USCT methods."""

from __future__ import annotations

from usctbench.registry import register_algorithm

from .refraction_gn import BentRayGNAdapter
from .rwave import RWaveAdapter


def register_adapter_algorithms(*, replace: bool = False) -> None:
    register_algorithm(
        "bent_ray_gn",
        BentRayGNAdapter,
        description="Refraction-corrected travel-time GN adapter with native smoke backend and optional MATLAB path.",
        tags=("adapter", "travel-time", "refraction"),
        replace=replace,
    )
    register_algorithm(
        "rwave_adapter",
        RWaveAdapter,
        description="Ray-Born/r-Wave adapter with native smoke backend and optional MATLAB path.",
        tags=("adapter", "ray-born", "rwave"),
        replace=replace,
    )


__all__ = ["BentRayGNAdapter", "RWaveAdapter", "register_adapter_algorithms"]
