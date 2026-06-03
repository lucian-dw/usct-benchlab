"""Wavefield-derived feature extraction.

The current main benchmark split keeps traditional ray methods on travel-time
surrogate cases and reserves k-Wave data for FWI.  The legacy extractor remains
available for debugging raw wavefield cases, but the retired channelized
apparent/eikonal/complex feature entry points are no longer exported.
"""

from .quality import extract_wavefield_features

__all__ = [
    "extract_wavefield_features",
]
