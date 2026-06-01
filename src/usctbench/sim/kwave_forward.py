"""k-Wave-compatible forward simulation entry points.

The default backend is a deterministic smoke-size wavefield generator used to
validate cache/QC/feature plumbing without requiring CUDA or a full k-Wave
install on the local Mac. Formal high-fidelity runs should set metadata/backend
to an actual k-Wave execution path on A100.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from usctbench.algorithms.ray.straight_projector import StraightRayProjector
from usctbench.data.synthetic import circular_attenuation, make_sound_speed_case
from usctbench.features.phase_delay import frequency_response_from_time
from usctbench.io.hdf5 import read_case_hdf5, write_case_hdf5
from usctbench.provenance import MeasurementProvenance, stamp_measurement_metadata
from usctbench.schema import GroundTruthSpec, MeasurementSpec, USCTCase
from usctbench.sim.cache import cached_case_matches, kwave_cache_key

_ENV_DEFAULT_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]*)\}")


def run_kwave_simulation_from_config(config_path: str | Path) -> Path:
    """Load a YAML config, simulate or reuse a matching wavefield case, and write it."""

    config_file = Path(config_path)
    config = _load_yaml(config_file)
    property_case = _load_or_make_property_case(config)
    simulation = dict(config.get("simulation", {}))
    simulation.setdefault("backend", config.get("backend", "native_smoke"))
    cache_key = kwave_cache_key(property_case, simulation)
    output_path = Path(_expand(str(config.get("outputs", {}).get("wavefield_case", "kwave_wavefield.h5"))))
    if output_path.exists():
        existing = read_case_hdf5(output_path)
        if cached_case_matches(existing, cache_key):
            return output_path
    wave_case = simulate_kwave_forward(property_case, simulation=simulation, config_name=str(config.get("name", config_file.stem)), cache_key=cache_key)
    write_case_hdf5(wave_case, output_path)
    return output_path


def simulate_kwave_forward(
    property_case: USCTCase,
    *,
    simulation: dict[str, Any],
    config_name: str = "kwave_simulation",
    cache_key: str | None = None,
) -> USCTCase:
    """Generate a smoke-size wavefield case from a property-map case."""

    if property_case.ground_truth.sound_speed_mps is None:
        raise ValueError("k-Wave simulation requires ground_truth.sound_speed_mps")
    c0 = float(simulation.get("reference_sound_speed_mps", property_case.metadata.get("reference_sound_speed_mps", 1500.0)))
    if c0 <= 0:
        raise ValueError("reference_sound_speed_mps must be positive")
    dt_s = float(simulation.get("dt_s", 8.0e-8))
    n_time = int(simulation.get("n_time", 768))
    if dt_s <= 0 or n_time < 8:
        raise ValueError("simulation dt_s must be positive and n_time must be >= 8")
    peak_frequency_hz = float(simulation.get("source_peak_frequency_hz", 250_000.0))
    frequencies_hz = np.asarray(simulation.get("frequencies_hz", [150_000.0, 250_000.0, 350_000.0, 450_000.0]), dtype=float)
    time_axis = np.arange(n_time, dtype=float) * dt_s

    projector = StraightRayProjector.from_case(property_case)
    sound_speed = np.asarray(property_case.ground_truth.sound_speed_mps, dtype=float)
    delta_slowness = (1.0 / sound_speed) - (1.0 / c0)
    delta_tof = projector.forward(delta_slowness).reshape(projector.ray_shape)
    distance = _pairwise_distance(property_case)
    water_tof = distance / c0
    case_tof = np.maximum(0.0, water_tof + delta_tof)
    attenuation = property_case.ground_truth.attenuation_np_per_m
    if attenuation is not None:
        attenuation_integral = projector.forward(np.asarray(attenuation, dtype=float)).reshape(projector.ray_shape)
    else:
        attenuation_integral = np.zeros(projector.ray_shape, dtype=float)
    geometric_spread = 1.0 / np.sqrt(np.maximum(distance, float(simulation.get("min_distance_m", 1.0e-3))))
    amplitude = geometric_spread * np.exp(-attenuation_integral)
    water_amplitude = geometric_spread
    valid = np.ones(projector.ray_shape, dtype=bool)
    if projector.ray_shape[0] == projector.ray_shape[1]:
        valid &= ~np.eye(projector.ray_shape[0], dtype=bool)

    time_data = _shifted_ricker_bank(time_axis, case_tof, amplitude, peak_frequency_hz, valid)
    water_reference = _shifted_ricker_bank(time_axis, water_tof, water_amplitude, peak_frequency_hz, valid)
    source_wavelet = _ricker(time_axis - 4.0 / peak_frequency_hz, peak_frequency_hz)
    freq_data = frequency_response_from_time(time_data, time_axis, frequencies_hz)
    metadata = _simulation_metadata(
        property_case,
        simulation=simulation,
        config_name=config_name,
        cache_key=cache_key,
        c0=c0,
        frequencies_hz=frequencies_hz,
    )
    return property_case.model_copy(
        update={
            "case_id": f"{property_case.case_id}_kwave_wavefield",
            "measurement": MeasurementSpec(
                domain="time",
                frequencies_hz=frequencies_hz,
                freq_data=freq_data,
                time_data=time_data,
                water_reference=water_reference,
                source_wavelet=source_wavelet,
                time_axis_s=time_axis,
                valid_mask=valid,
            ),
            "metadata": metadata,
        }
    )


def _load_or_make_property_case(config: dict[str, Any]) -> USCTCase:
    case_config = dict(config.get("case", {}))
    case_path = case_config.get("path")
    if case_path:
        return read_case_hdf5(_expand(str(case_path)))
    shape_value = case_config.get("shape", [48, 48])
    shape = (int(shape_value[0]), int(shape_value[1]))
    n_transducers = int(case_config.get("n_transducers", 16))
    case = make_sound_speed_case(
        case_id=str(case_config.get("case_id", "kwave_smoke_property")),
        shape=shape,
        n_transducers=n_transducers,
        background_mps=float(case_config.get("background_mps", 1500.0)),
        inclusion_mps=float(case_config.get("inclusion_mps", 1450.0)),
    )
    attenuation = circular_attenuation(
        case.grid,
        inclusion_np_per_m=float(case_config.get("inclusion_attenuation_np_per_m", 3.0)),
        radius_m=float(case_config.get("inclusion_radius_m", 0.006)),
    )
    density = np.full(case.grid.shape, float(case_config.get("density_kg_per_m3", 1000.0)), dtype=float)
    return case.model_copy(
        update={
            "ground_truth": GroundTruthSpec(
                sound_speed_mps=case.ground_truth.sound_speed_mps,
                attenuation_np_per_m=attenuation,
                density_kg_per_m3=density,
            ),
            "metadata": {
                **case.metadata,
                "case_type": "property_map",
                "property_map_source": "synthetic_smoke",
                "density_kg_per_m3_default": float(case_config.get("density_kg_per_m3", 1000.0)),
            },
        }
    )


def _simulation_metadata(
    property_case: USCTCase,
    *,
    simulation: dict[str, Any],
    config_name: str,
    cache_key: str | None,
    c0: float,
    frequencies_hz: np.ndarray,
) -> dict[str, Any]:
    backend = str(simulation.get("backend", "native_smoke"))
    simulation_metadata = {
        "config_name": config_name,
        "backend": backend,
        "cache_key": cache_key,
        "grid_shape": list(property_case.grid.shape),
        "spacing_m": list(property_case.grid.spacing_m),
        "n_tx": int(property_case.geometry.tx_pos_m.shape[0]),
        "n_rx": int(property_case.geometry.rx_pos_m.shape[0]),
        "reference_sound_speed_mps": c0,
        "source_wavelet": simulation.get("source_wavelet", "ricker"),
        "source_peak_frequency_hz": float(simulation.get("source_peak_frequency_hz", 250_000.0)),
        "dt_s": float(simulation.get("dt_s", 8.0e-8)),
        "n_time": int(simulation.get("n_time", 768)),
        "pml_thickness_pixels": int(simulation.get("pml_thickness_pixels", 8)),
        "frequencies_hz": [float(value) for value in frequencies_hz.tolist()],
    }
    return stamp_measurement_metadata(
        property_case.metadata,
        measurement_provenance=MeasurementProvenance.SELF_SIMULATED_KWAVE_WAVEFIELD,
        benchmark_type="self_simulated_kwave_wavefield",
        forward_model="k_wave_unified_wavefield_smoke_backend",
        feature_source="raw_time_and_frequency_wavefield",
        uses_complex_wavefield=True,
        extra={
            "case_type": "kwave_wavefield",
            "measurement_domain": "time",
            "simulation_backend": backend,
            "simulation_cache_key": cache_key,
            "simulation_metadata": simulation_metadata,
            "measurement_limitations": [
                "self-simulated wavefield benchmark; requires simulation QC before algorithm conclusions",
                "default native_smoke backend validates the unified data path and is not a replacement for high-fidelity A100 k-Wave runs",
            ],
        },
    )


def _pairwise_distance(case: USCTCase) -> np.ndarray:
    tx = np.asarray(case.geometry.tx_pos_m, dtype=float)
    rx = np.asarray(case.geometry.rx_pos_m, dtype=float)
    diff = tx[:, None, :] - rx[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def _shifted_ricker_bank(
    time_axis_s: np.ndarray,
    tof_s: np.ndarray,
    amplitude: np.ndarray,
    peak_frequency_hz: float,
    valid: np.ndarray,
) -> np.ndarray:
    shifted = time_axis_s[None, None, :] - np.asarray(tof_s, dtype=float)[..., None]
    data = np.asarray(amplitude, dtype=float)[..., None] * _ricker(shifted, peak_frequency_hz)
    return np.where(valid[..., None], data, 0.0)


def _ricker(t_s: np.ndarray, peak_frequency_hz: float) -> np.ndarray:
    arg = np.pi * float(peak_frequency_hz) * np.asarray(t_s, dtype=float)
    arg2 = arg**2
    return (1.0 - 2.0 * arg2) * np.exp(-arg2)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"simulation config must be a YAML mapping: {path}")
    return _expand_config_value(payload)


def _expand(value: str) -> str:
    def _replace_default(match: re.Match[str]) -> str:
        env_value = os.environ.get(match.group(1))
        return env_value if env_value not in (None, "") else match.group(2)

    value = _ENV_DEFAULT_RE.sub(_replace_default, value)
    return os.path.expandvars(os.path.expanduser(value))


def _expand_config_value(value: Any) -> Any:
    if isinstance(value, str):
        return _expand(value)
    if isinstance(value, list):
        return [_expand_config_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_expand_config_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _expand_config_value(item) for key, item in value.items()}
    return value
