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
    passed, reasons = _qc_pass_fail(metrics)
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
    spacing_min = min(float(value) for value in case.grid.spacing_m)
    wavelength_min = c_min / peak_frequency if peak_frequency > 0 else float("inf")
    points_per_wavelength = wavelength_min / spacing_min if spacing_min > 0 else 0.0
    cfl = c_max * dt / spacing_min / np.sqrt(2.0) if spacing_min > 0 else float("inf")
    energy = np.nansum(time_data**2, axis=-1)
    finite_energy = energy[np.isfinite(energy) & (energy > 0.0)]
    median_energy = float(np.median(finite_energy)) if finite_energy.size else 0.0
    bad_receiver = ~np.isfinite(energy) | (energy <= max(1.0e-18, median_energy * 1.0e-8))
    water_tof, water_valid = peak_tof(water, time_axis)
    water_geometry_tof = _pairwise_distance(case) / float(simulation.get("reference_sound_speed_mps", case.metadata.get("reference_sound_speed_mps", 1500.0)))
    water_tof_rmse = _rmse(water_tof, water_geometry_tof, water_valid)
    reciprocity = _reciprocity_error(time_data)
    boundary_fraction = _boundary_energy_fraction(time_data)
    feature_payload = extract_wavefield_features(case, method="all")[1]
    source_bandwidth = _source_bandwidth(case)
    nan_inf = int(np.sum(~np.isfinite(time_data)) + np.sum(~np.isfinite(water)))
    dynamic_range_db = _dynamic_range_db(finite_energy)
    return {
        "grid_points_per_wavelength_min": float(points_per_wavelength),
        "cfl_number": float(cfl),
        "pml_thickness_pixels": int(simulation.get("pml_thickness_pixels", 0)),
        "source_peak_frequency_hz": float(peak_frequency),
        "source_bandwidth_hz": float(source_bandwidth),
        "receiver_signal_energy_min": float(np.min(finite_energy)) if finite_energy.size else 0.0,
        "receiver_signal_energy_median": float(median_energy),
        "receiver_signal_energy_max": float(np.max(finite_energy)) if finite_energy.size else 0.0,
        "bad_receiver_fraction": float(np.mean(bad_receiver)) if bad_receiver.size else 1.0,
        "reciprocity_error": float(reciprocity),
        "water_tof_rmse_vs_geometry": float(water_tof_rmse),
        "phase_unwrap_failure_fraction": float(feature_payload.get("phase_unwrap_failure_fraction", 1.0)),
        "tof_valid_fraction": float(feature_payload.get("tof_valid_fraction", 0.0)),
        "amplitude_dynamic_range_db": float(dynamic_range_db),
        "nan_inf_count": nan_inf,
        "boundary_energy_fraction": float(boundary_fraction),
    }


def _qc_pass_fail(metrics: dict[str, Any]) -> tuple[bool, list[str]]:
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


def _write_qc_artifacts(case: USCTCase, out_dir: Path, metrics: dict[str, Any]) -> None:
    time_data = np.asarray(case.measurement.time_data, dtype=float)
    source = np.asarray(case.measurement.source_wavelet, dtype=float) if case.measurement.source_wavelet is not None else np.zeros(time_data.shape[-1])
    write_preview_png(np.nanmax(np.abs(time_data), axis=-1), out_dir / "wavefield_preview.png")
    write_preview_png(_spectrum_image(source, case.measurement.time_axis_s), out_dir / "source_spectrum.png")
    write_preview_png(_hist_image(np.nansum(time_data**2, axis=-1).reshape(-1)), out_dir / "receiver_energy_hist.png")
    write_preview_png(np.asarray([[metrics["water_tof_rmse_vs_geometry"]]], dtype=float), out_dir / "water_tof_error.png")
    write_preview_png(_reciprocity_image(time_data), out_dir / "reciprocity_error.png")
    write_preview_png(_boundary_energy_map(time_data), out_dir / "boundary_energy.png")
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


def _reciprocity_error(time_data: np.ndarray) -> float:
    if time_data.shape[0] != time_data.shape[1]:
        return 0.0
    diff = time_data - np.swapaxes(time_data, 0, 1)
    denom = float(np.linalg.norm(time_data))
    return float(np.linalg.norm(diff) / denom) if denom > 0.0 else 0.0


def _reciprocity_image(time_data: np.ndarray) -> np.ndarray:
    if time_data.shape[0] != time_data.shape[1]:
        return np.zeros(time_data.shape[:2], dtype=float)
    return np.sqrt(np.nanmean((time_data - np.swapaxes(time_data, 0, 1)) ** 2, axis=-1))


def _boundary_energy_fraction(time_data: np.ndarray) -> float:
    n = time_data.shape[-1]
    width = max(1, n // 10)
    boundary = np.nansum(time_data[..., :width] ** 2) + np.nansum(time_data[..., -width:] ** 2)
    total = np.nansum(time_data**2)
    return float(boundary / total) if total > 0.0 else 1.0


def _boundary_energy_map(time_data: np.ndarray) -> np.ndarray:
    n = time_data.shape[-1]
    width = max(1, n // 10)
    boundary = np.nansum(time_data[..., :width] ** 2, axis=-1) + np.nansum(time_data[..., -width:] ** 2, axis=-1)
    total = np.nansum(time_data**2, axis=-1)
    return np.divide(boundary, total, out=np.zeros_like(boundary), where=total > 0.0)


def _dynamic_range_db(energy: np.ndarray) -> float:
    finite = np.asarray(energy, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0.0)]
    if finite.size == 0:
        return 0.0
    return float(10.0 * np.log10(float(np.max(finite)) / float(np.min(finite))))
