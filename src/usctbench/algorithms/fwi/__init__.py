"""Tiny waveform-inversion proof-of-life components."""

from __future__ import annotations

from usctbench.core.registry import register_algorithm

from .adapter import KWaveFWIAdapterAlgorithm
from .tiny import TinyFWIAlgorithm


def register_fwi_algorithms(*, replace: bool = False) -> None:
    register_algorithm(
        "fwi_tiny",
        TinyFWIAlgorithm,
        description="Tiny synthetic waveform-inversion proof-of-life.",
        tags=("fwi", "synthetic", "sound-speed"),
        replace=replace,
    )
    register_algorithm(
        "fwi_kwave_adapter",
        KWaveFWIAdapterAlgorithm,
        description="Adapter for k-Wave/WaveformInversionUST MATLAB FWI results.",
        tags=("fwi", "kwave", "external"),
        replace=replace,
    )


__all__ = ["KWaveFWIAdapterAlgorithm", "TinyFWIAlgorithm", "register_fwi_algorithms"]
