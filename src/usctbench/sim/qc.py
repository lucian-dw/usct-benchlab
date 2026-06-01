"""Simulation quality control for wavefield benchmark cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from usctbench.features.quality import extract_wavefield_features
from usctbench.features.tof import peak_tof
from usctbench.io.hdf5 import read_case_hdf5, write_case_hdf5
from usctbench.schema import USCTCase
from usctbench.viz.preview import write_preview_png


def run_simulation_qc(case: USCTCase | str | Path, out_dir: str | Path | None = None, *, update_case: bool = True) -> dict[str, Any]:
    """Compute QC metrics/artifacts and optionally stamp the case metadata."""

    case_path = Path(case) if isinstance(case, (str, Path)) else None
    wave_case = read_case_hdf5(case_path) if case_path is not None else case
    output_dir = Path(out_dir) if out_dir is not None else (case_path.parent if case_path is not None else Path.cwd())
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = simulation_qc_metrics(wave_case)
    passed, reasons = _qc_pass_fail(metrics, wave_case)
    payload = {"passed": passed, "fail_reasons": reasons, "metrics": metrics}
    (output_dir / "simulation_qc.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_qc_artifacts(wave_case, output_dir, metrics)
    if update_case:
        stamped = wave_case.model_copy(
            update={
                "metadata": {
                    **wave_case.metadata,
                    "simulation_qc_passed": bool(passed),
                    "simulation_failed_qc": not bool(passed),
                    "simulation_qc": payload,
                }
            }
        )
        if case_path is not None:
            write_case_hdf5(stamped, case_path)
        wave_case = stamped
    return payload


def simulation_qc_metrics(case: USCTCase) -> dict[str, Any]:
    """Return required simulation QC metrics for a wavefield case."""

    if case.measurement.time_data is None or case.measurement.water_reference is None or case.measurement.time_axis_s is None:
        raise ValueError("simulation QC requires time_data, water_reference, and time_axis_s")
    time_data = np.asarray(case.measurement.time_data, dtype=float)
    water = np.asarray(case.measurement.water_reference, dtype=float)
    time_axis = np.asarray(case.measurement.time_axis_s, dtype=float)
    simulation = dict(case.metadata.get("simulation_metadata", {}))
    dt = float(np.median(np.diff(time_axis))) if time_axis.size > 1 else float(simulation.get("dt_s", 0.0))
    c_min, c_max = _sound_speed_range(case)
    peak_frequency = float(simulation.get("source_peak_frequency_hz", _source_peak_frequency(case)))
    frequencies = np.asarray(case.measurement.frequencies_hz, dtype=float).reshape(-1) if case.measurement.frequencies_hz is not None else np.asarray([], dtype=float)
    effective_max_frequency = max([peak_frequency, *(float(value) for value in frequencies if np.isfinite(value))])
    spacing_min = min(float(value) for value in case.grid.spacing_m)
    wavelength_min = c_min / effective_max_frequency if effective_max_frequency > 0 else float("inf")
    points_per_wavelength = wavelength_min / spacing_min if spacing_min > 0 else 0.0
    cfl = c_max * dt / spacing_min / np.sqrt(2.0) if spacing_min > 0 else float("inf")
    valid_trace_mask = _valid_trace_mask(case)
    energy = np.nansum(time_data**2, axis=-1)
    finite_energy = energy[valid_trace_mask & np.isfinite(energy) & (energy > 0.0)]
    median_energy = float(np.median(finite_energy)) if finite_energy.size else 0.0
    bad_receiver = (~np.isfinite(energy) | (energy <= max(1.0e-18, median_energy * 1.0e-8))) & valid_trace_mask
    valid_trace_count = int(np.sum(valid_trace_mask))
    bad_receiver_fraction = float(np.sum(bad_receiver) / valid_trace_count) if valid_trace_count else 1.0
    water_tof, water_valid = peak_tof(water, time_axis)
    water_valid = water_valid & valid_trace_mask
    water_geometry_tof = _pairwise_distance(case) / float(simulation.get("reference_sound_speed_mps", case.metadata.get("reference_sound_speed_mps", 1500.0)))
    water_tof_raw_rmse = _rmse(water_tof, water_geometry_tof, water_valid)
    water_tof_bias = _median_error(water_tof, water_geometry_tof, water_valid)
    water_tof_aligned = water_tof - (water_tof_bias if np.isfinite(water_tof_bias) else 0.0)
    water_tof_rmse = _rmse(water_tof_aligned, water_geometry_tof, water_valid)
    reciprocity = _reciprocity_error(time_data, valid_trace_mask)
    boundary_fraction = _boundary_energy_fraction(time_data, valid_trace_mask)
    feature_payload = extract_wavefield_features(case, method="all")[1]
    source_bandwidth = _source_bandwidth(case)
    nan_inf = int(np.sum(~np.isfinite(time_data)) + np.sum(~np.isfinite(water)))
    dynamic_range_db = _dynamic_range_db(finite_energy)
    return {
        "grid_points_per_wavelength_min": float(points_per_wavelength),
        "cfl_number": float(cfl),
        "pml_thickness_pixels": int(simulation.get("pml_thickness_pixels", 0)),
        "source_peak_frequency_hz": float(peak_frequency),
        "effective_max_frequency_hz": float(effective_max_frequency),
        "source_bandwidth_hz": float(source_bandwidth),
        "receiver_signal_energy_min": float(np.min(finite_energy)) if finite_energy.size else 0.0,
        "receiver_signal_energy_median": float(median_energy),
        "receiver_signal_energy_max": float(np.max(finite_energy)) if finite_energy.size else 0.0,
        "bad_receiver_fraction": bad_receiver_fraction,
        "excluded_receiver_fraction": float(1.0 - np.mean(valid_trace_mask)) if valid_trace_mask.size else 1.0,
        "reciprocity_error": float(reciprocity),
        "water_tof_rmse_vs_geometry": float(water_tof_rmse),
        "water_tof_raw_rmse_vs_geometry": float(water_tof_raw_rmse),
        "water_tof_bias_s": float(water_tof_bias),
        "phase_unwrap_failure_fraction": float(feature_payload.get("phase_unwrap_failure_fraction", 1.0)),
        "tof_valid_fraction": float(feature_payload.get("tof_valid_fraction", 0.0)),
        "amplitude_dynamic_range_db": float(dynamic_range_db),
        "nan_inf_count": nan_inf,
        "boundary_energy_fraction": float(boundary_fraction),
        "dt_s": float(dt),
        "simulation_backend": str(simulation.get("backend", case.metadata.get("simulation_backend", ""))),
    }


def _qc_pass_fail(metrics: dict[str, Any], case: USCTCase) -> tuple[bool, list[str]]:
    backend = str(metrics.get("simulation_backend", case.metadata.get("simulation_backend", ""))).lower()
    is_real_kwave = backend not in {"native_smoke", "smoke", "analytic_smoke", ""}
    if not is_real_kwave:
        checks = {
            "grid_points_per_wavelength_min": float(metrics.get("grid_points_per_wavelength_min", 0.0)) >= 3.0,
            "cfl_number": float(metrics.get("cfl_number", float("inf"))) <= 0.5,
            "bad_receiver_fraction": float(metrics.get("bad_receiver_fraction", 1.0)) <= 0.2,
            "reciprocity_error": float(metrics.get("reciprocity_error", 1.0)) <= 0.35,
            "tof_valid_fraction": float(metrics.get("tof_valid_fraction", 0.0)) >= 0.5,
            "nan_inf_count": int(metrics.get("nan_inf_count", 1)) == 0,
            "boundary_energy_fraction": float(metrics.get("boundary_energy_fraction", 1.0)) <= 0.4,
        }
        reasons = [f"{name} failed" for name, ok in checks.items() if not ok]
        return not reasons, reasons

    config_name = str(case.metadata.get("simulation_metadata", {}).get("config_name", "")).lower()
    ppw_min = 8.0 if "quality" in config_name else 6.0
    dt = float(metrics.get("dt_s", float("nan")))
    water_limit = 0.5 * dt if np.isfinite(dt) and dt > 0.0 else 0.0
    n_tx = max(1, int(case.geometry.tx_pos_m.shape[0]))
    expected_self_pair_fraction = 1.0 / float(n_tx) if n_tx == int(case.geometry.rx_pos_m.shape[0]) else 0.0
    bad_receiver_limit = max(0.15, expected_self_pair_fraction + 0.05)
    checks = {
        f"grid_points_per_wavelength_min >= {ppw_min:g}": float(metrics.get("grid_points_per_wavelength_min", 0.0)) >= ppw_min,
        "cfl_number <= 0.3": float(metrics.get("cfl_number", float("inf"))) <= 0.3,
        "pml_thickness_pixels >= 20": int(metrics.get("pml_thickness_pixels", 0)) >= 20,
        f"bad_receiver_fraction <= {bad_receiver_limit:g}": float(metrics.get("bad_receiver_fraction", 1.0)) <= bad_receiver_limit,
        "reciprocity_error <= 0.1": float(metrics.get("reciprocity_error", 1.0)) <= 0.1,
        f"water_tof_rmse_vs_geometry <= 0.5*dt ({water_limit:.3g}s)": bool(water_limit > 0.0)
        and float(metrics.get("water_tof_rmse_vs_geometry", float("inf"))) <= water_limit,
        "tof_valid_fraction >= 0.75": float(metrics.get("tof_valid_fraction", 0.0)) >= 0.75,
        "nan_inf_count": int(metrics.get("nan_inf_count", 1)) == 0,
        "boundary_energy_fraction": float(metrics.get("boundary_energy_fraction", 1.0)) <= 0.4,
    }
    reasons = [f"{name} failed" for name, ok in checks.items() if not ok]
    return not reasons, reasons


def _write_qc_artifacts(case: USCTCase, out_dir: Path, metrics: dict[str, Any]) -> None:
    time_data = np.asarray(case.measurement.time_data, dtype=float)
    valid_trace_mask = _valid_trace_mask(case)
    source = np.asarray(case.measurement.source_wavelet, dtype=float) if case.measurement.source_wavelet is not None else np.zeros(time_data.shape[-1])
    write_preview_png(np.nanmax(np.abs(time_data), axis=-1), out_dir / "wavefield_preview.png")
    write_preview_png(_spectrum_image(source, case.measurement.time_axis_s), out_dir / "source_spectrum.png")
    write_preview_png(_hist_image(np.nansum(time_data**2, axis=-1)[valid_trace_mask].reshape(-1)), out_dir / "receiver_energy_hist.png")
    write_preview_png(np.asarray([[metrics["water_tof_rmse_vs_geometry"]]], dtype=float), out_dir / "water_tof_error.png")
    write_preview_png(_reciprocity_image(time_data, valid_trace_mask), out_dir / "reciprocity_error.png")
    write_preview_png(_boundary_energy_map(time_data, valid_trace_mask), out_dir / "boundary_energy.png")
    try:
        feature_case, _ = extract_wavefield_features(case)
        if feature_case.measurement.feature_quality is not None:
            write_preview_png(feature_case.measurement.feature_quality, out_dir / "feature_preview.png")
    except Exception:
        write_preview_png(np.zeros(time_data.shape[:2], dtype=float), out_dir / "feature_preview.png")


def _sound_speed_range(case: USCTCase) -> tuple[float, float]:
    values = case.ground_truth.sound_speed_mps
    if values is None:
        c0 = float(case.metadata.get("reference_sound_speed_mps", 1500.0))
        return c0, c0
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1500.0, 1500.0
    return float(np.min(finite)), float(np.max(finite))


def _source_peak_frequency(case: USCTCase) -> float:
    source = case.measurement.source_wavelet
    time_axis = case.measurement.time_axis_s
    if source is None or time_axis is None or len(time_axis) < 2:
        return 0.0
    freqs, spectrum = _spectrum(source, time_axis)
    if freqs.size == 0:
        return 0.0
    return float(freqs[int(np.argmax(spectrum))])


def _source_bandwidth(case: USCTCase) -> float:
    source = case.measurement.source_wavelet
    time_axis = case.measurement.time_axis_s
    if source is None or time_axis is None or len(time_axis) < 2:
        return 0.0
    freqs, spectrum = _spectrum(source, time_axis)
    if spectrum.size == 0 or float(np.max(spectrum)) <= 0.0:
        return 0.0
    keep = spectrum >= 0.5 * float(np.max(spectrum))
    return float(np.max(freqs[keep]) - np.min(freqs[keep])) if np.any(keep) else 0.0


def _spectrum(source: np.ndarray, time_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dt = float(np.median(np.diff(np.asarray(time_axis, dtype=float))))
    freqs = np.fft.rfftfreq(np.asarray(source).size, d=dt)
    spectrum = np.abs(np.fft.rfft(np.asarray(source, dtype=float)))
    return freqs, spectrum


def _spectrum_image(source: np.ndarray, time_axis: np.ndarray | None) -> np.ndarray:
    if time_axis is None or np.asarray(source).size < 2:
        return np.zeros((64, 128), dtype=float)
    _, spectrum = _spectrum(source, np.asarray(time_axis, dtype=float))
    return _line_image(spectrum[:128])


def _hist_image(values: np.ndarray) -> np.ndarray:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.zeros((64, 128), dtype=float)
    counts, _ = np.histogram(finite, bins=128)
    return _line_image(counts)


def _line_image(values: np.ndarray, *, height: int = 64) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return np.zeros((height, 1), dtype=float)
    values = values - float(np.min(values))
    denom = float(np.max(values))
    if denom > 0.0:
        values = values / denom
    image = np.zeros((height, values.size), dtype=float)
    rows = np.clip((height - 1 - np.rint(values * (height - 1))).astype(int), 0, height - 1)
    image[rows, np.arange(values.size)] = 1.0
    return image


def _pairwise_distance(case: USCTCase) -> np.ndarray:
    tx = np.asarray(case.geometry.tx_pos_m, dtype=float)
    rx = np.asarray(case.geometry.rx_pos_m, dtype=float)
    return np.linalg.norm(tx[:, None, :] - rx[None, :, :], axis=-1)


def _rmse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    finite = np.asarray(mask, dtype=bool) & np.isfinite(a) & np.isfinite(b)
    if not np.any(finite):
        return float("nan")
    diff = np.asarray(a, dtype=float)[finite] - np.asarray(b, dtype=float)[finite]
    return float(np.sqrt(np.mean(diff**2)))


def _median_error(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    finite = np.asarray(mask, dtype=bool) & np.isfinite(a) & np.isfinite(b)
    if not np.any(finite):
        return float("nan")
    diff = np.asarray(a, dtype=float)[finite] - np.asarray(b, dtype=float)[finite]
    return float(np.median(diff))


def _valid_trace_mask(case: USCTCase) -> np.ndarray:
    if case.measurement.time_data is None:
        shape = (case.geometry.tx_pos_m.shape[0], case.geometry.rx_pos_m.shape[0])
    else:
        shape = tuple(np.asarray(case.measurement.time_data).shape[:2])
    mask = np.ones(shape, dtype=bool)
    if case.measurement.valid_mask is not None:
        configured = np.asarray(case.measurement.valid_mask, dtype=bool)
        if configured.shape == shape:
            mask &= configured
    if shape[0] == shape[1]:
        mask &= ~np.eye(shape[0], dtype=bool)
    return mask


def _reciprocity_error(time_data: np.ndarray, trace_mask: np.ndarray | None = None) -> float:
    if time_data.shape[0] != time_data.shape[1]:
        return 0.0
    if trace_mask is not None:
        mask = np.asarray(trace_mask, dtype=bool)
        pair_mask = mask & mask.T
        if not np.any(pair_mask):
            return 1.0
        diff = (time_data - np.swapaxes(time_data, 0, 1))[pair_mask]
        denom_values = time_data[pair_mask]
    else:
        diff = time_data - np.swapaxes(time_data, 0, 1)
        denom_values = time_data
    denom = float(np.linalg.norm(denom_values))
    return float(np.linalg.norm(diff) / denom) if denom > 0.0 else 0.0


def _reciprocity_image(time_data: np.ndarray, trace_mask: np.ndarray | None = None) -> np.ndarray:
    if time_data.shape[0] != time_data.shape[1]:
        return np.zeros(time_data.shape[:2], dtype=float)
    image = np.sqrt(np.nanmean((time_data - np.swapaxes(time_data, 0, 1)) ** 2, axis=-1))
    if trace_mask is not None:
        image = np.where(np.asarray(trace_mask, dtype=bool), image, 0.0)
    return image


def _boundary_energy_fraction(time_data: np.ndarray, trace_mask: np.ndarray | None = None) -> float:
    n = time_data.shape[-1]
    width = max(1, n // 10)
    traces = np.asarray(time_data, dtype=float)
    if trace_mask is not None:
        mask = np.asarray(trace_mask, dtype=bool)
        traces = traces[mask]
    boundary = np.nansum(traces[..., :width] ** 2) + np.nansum(traces[..., -width:] ** 2)
    total = np.nansum(traces**2)
    return float(boundary / total) if total > 0.0 else 1.0


def _boundary_energy_map(time_data: np.ndarray, trace_mask: np.ndarray | None = None) -> np.ndarray:
    n = time_data.shape[-1]
    width = max(1, n // 10)
    boundary = np.nansum(time_data[..., :width] ** 2, axis=-1) + np.nansum(time_data[..., -width:] ** 2, axis=-1)
    total = np.nansum(time_data**2, axis=-1)
    image = np.divide(boundary, total, out=np.zeros_like(boundary), where=total > 0.0)
    if trace_mask is not None:
        image = np.where(np.asarray(trace_mask, dtype=bool), image, 0.0)
    return image


def _dynamic_range_db(energy: np.ndarray) -> float:
    finite = np.asarray(energy, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0.0)]
    if finite.size == 0:
        return 0.0
    return float(10.0 * np.log10(float(np.max(finite)) / float(np.min(finite))))
