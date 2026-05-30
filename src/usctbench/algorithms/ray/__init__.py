"""Straight-ray reconstruction baselines."""

from __future__ import annotations

from usctbench.registry import register_algorithm

from .attenuation import AttenuationSIRTAlgorithm
from .cgls import StraightRayCGLSAlgorithm
from .sart import StraightRaySARTAlgorithm
from .sirt import StraightRaySIRTAlgorithm


def register_ray_algorithms(*, replace: bool = False) -> None:
    """Register built-in ray algorithms."""

    register_algorithm(
        "straight_sart",
        StraightRaySARTAlgorithm,
        description="Straight-ray SART sound-speed reconstruction.",
        tags=("ray", "sound-speed"),
        replace=replace,
    )
    register_algorithm(
        "straight_sirt",
        StraightRaySIRTAlgorithm,
        description="Straight-ray SIRT sound-speed reconstruction.",
        tags=("ray", "sound-speed"),
        replace=replace,
    )
    register_algorithm(
        "straight_cgls",
        StraightRayCGLSAlgorithm,
        description="Straight-ray CGLS sound-speed reconstruction.",
        tags=("ray", "sound-speed"),
        replace=replace,
    )
    register_algorithm(
        "attenuation_sirt",
        AttenuationSIRTAlgorithm,
        description="Straight-ray SIRT attenuation reconstruction.",
        tags=("ray", "attenuation"),
        replace=replace,
    )


__all__ = [
    "AttenuationSIRTAlgorithm",
    "StraightRayCGLSAlgorithm",
    "StraightRaySARTAlgorithm",
    "StraightRaySIRTAlgorithm",
    "register_ray_algorithms",
]

