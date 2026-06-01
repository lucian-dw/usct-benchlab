"""Simulation cache keys and metadata matching."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from usctbench.schema import USCTCase


def kwave_cache_key(case: USCTCase, simulation: dict[str, Any]) -> str:
    """Return a stable cache key for a case/config simulation request."""

    payload = {
        "case_id": case.case_id,
        "grid_shape": list(case.grid.shape),
        "grid_spacing_m": list(case.grid.spacing_m),
        "n_tx": int(case.geometry.tx_pos_m.shape[0]),
        "n_rx": int(case.geometry.rx_pos_m.shape[0]),
        "array_mode": simulation.get("array_mode"),
        "source_wavelet": simulation.get("source_wavelet", "ricker"),
        "source_peak_frequency_hz": simulation.get("source_peak_frequency_hz"),
        "source_bandwidth_hz": simulation.get("source_bandwidth_hz"),
        "frac_bw": simulation.get("frac_bw"),
        "pml_thickness_pixels": simulation.get("pml_thickness_pixels"),
        "pml_size": simulation.get("pml_size"),
        "ncalc": simulation.get("ncalc"),
        "xmax_mm": simulation.get("xmax_mm"),
        "cfl_number": simulation.get("cfl_number"),
        "downsample_factor": simulation.get("downsample_factor"),
        "dt_s": simulation.get("dt_s"),
        "n_time": simulation.get("n_time"),
        "frequencies_hz": simulation.get("frequencies_hz"),
        "backend": simulation.get("backend"),
        "sound_speed_digest": _array_digest(case.ground_truth.sound_speed_mps),
        "attenuation_digest": _array_digest(case.ground_truth.attenuation_np_per_m),
        "density_digest": _array_digest(case.ground_truth.density_kg_per_m3),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cached_case_matches(case: USCTCase, cache_key: str) -> bool:
    """Return true when an existing wavefield case matches a simulation key."""

    metadata = case.metadata.get("simulation_metadata", {})
    return str(metadata.get("cache_key", case.metadata.get("simulation_cache_key", ""))) == str(cache_key)


def _array_digest(value: np.ndarray | None) -> str | None:
    if value is None:
        return None
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()
