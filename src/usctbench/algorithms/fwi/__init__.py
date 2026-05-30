"""Tiny waveform-inversion proof-of-life components."""

from __future__ import annotations

from usctbench.registry import register_algorithm

from .tiny_fwi import TinyFWIAlgorithm


def register_fwi_algorithms(*, replace: bool = False) -> None:
    register_algorithm(
        "fwi_tiny",
        TinyFWIAlgorithm,
        description="Tiny synthetic waveform-inversion proof-of-life.",
        tags=("fwi", "synthetic", "sound-speed"),
        replace=replace,
    )


__all__ = ["TinyFWIAlgorithm", "register_fwi_algorithms"]

