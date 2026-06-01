"""Feature extraction orchestration and quality summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from usctbench.features.amplitude import log_amplitude_ratio_from_frequency, log_amplitude_ratio_from_time
from usctbench.features.phase_delay import frequency_response_from_time, multi_frequency_phase_slope_delay
from usctbench.features.tof import cross_correlation_delay, first_arrival_tof
from usctbench.io.hdf5 import read_case_hdf5, write_case_hdf5
from usctbench.provenance import MeasurementProvenance, stamp_measurement_metadata
from usctbench.schema import MeasurementSpec, USCTCase
from usctbench.viz.preview import write_preview_png


def extract_wavefield_features(
    case: USCTCase | str | Path,
    *,
    out: str | Path | None = None,
    method: str = "all",
    min_phase_frequencies: int = 3,
) -> tuple[USCTCase, dict[str, Any]]:
    """Extract ToF, phase-delay, amplitude, masks, and QC from a wavefield case."""

    wave_case = read_case_hdf5(case) if isinstance(case, (str, Path)) else case
    if wave_case.measurement.time_data is None:
        raise ValueError("wavefield feature extraction requires measurement.time_data")
    if wave_case.measurement.water_reference is None:
        raise ValueError("wavefield feature extraction requires measurement.water_reference")
    if wave_case.measurement.time_axis_s is None:
        raise ValueError("wavefield feature extraction requires measurement.time_axis_s")

    time_data = np.asarray(wave_case.measurement.time_data, dtype=float)
    reference = np.asarray(wave_case.measurement.water_reference, dtype=float)
    time_axis = np.asarray(wave_case.measurement.time_axis_s, dtype=float)
    first_arrival, first_valid = first_arrival_tof(time_data, time_axis)
    water_first, water_first_valid = first_arrival_tof(reference, time_axis)
    xcorr_delay, xcorr_valid = cross_correlation_delay(time_data, reference, time_axis)

    frequencies = wave_case.measurement.frequencies_hz
    if frequencies is None:
        frequencies = np.asarray(wave_case.metadata.get("simulation_metadata", {}).get("frequencies_hz", []), dtype=float)
    frequencies = np.asarray(frequencies, dtype=float).reshape(-1)
    if frequencies.size:
        freq_data = wave_case.measurement.freq_data
        if freq_data is None:
            freq_data = frequency_response_from_time(time_data, time_axis, frequencies)
        reference_freq = frequency_response_from_time(reference, time_axis, frequencies)
        phase_delay, phase_rms, phase_valid = multi_frequency_phase_slope_delay(
            freq_data,
            reference_freq,
            frequencies,
            min_frequencies=min_phase_frequencies,
        )
        log_amp = log_amplitude_ratio_from_frequency(freq_data, reference_freq)
        log_amp_for_solver = np.nanmean(log_amp, axis=0)
    else:
        freq_data = None
        phase_delay = np.full(time_data.shape[:2], np.nan, dtype=float)
        phase_rms = np.full(time_data.shape[:2], np.inf, dtype=float)
        phase_valid = np.zeros(time_data.shape[:2], dtype=bool)
        log_amp_for_solver = log_amplitude_ratio_from_time(time_data, reference)

    first_delta = first_arrival - water_first
    base_valid = np.isfinite(xcorr_delay) & xcorr_valid & first_valid & water_first_valid
    non_self = _non_self_mask(wave_case)
    valid_mask = base_valid & non_self
    feature_quality = _feature_quality_map(
        first_delta=first_delta,
        xcorr_delay=xcorr_delay,
        phase_delay=phase_delay,
        phase_valid=phase_valid,
        valid_mask=valid_mask,
    )
    if method == "phase-slope":
        selected_delta = phase_delay
        selected_valid = valid_mask & phase_valid
        selected_source = "phase_slope_delay"
    else:
        selected_delta = xcorr_delay
        selected_valid = valid_mask
        selected_source = "xcorr_tof_delay"

    qc = _feature_qc(
        first_delta=first_delta,
        xcorr_delay=xcorr_delay,
        phase_delay=phase_delay,
        phase_valid=phase_valid,
        valid_mask=selected_valid,
        log_amp=log_amp_for_solver,
    )
    provenance = wave_case.metadata.get("measurement_provenance", MeasurementProvenance.SELF_SIMULATED_KWAVE_WAVEFIELD.value)
    metadata = stamp_measurement_metadata(
        wave_case.metadata,
        measurement_provenance=provenance,
        benchmark_type=str(wave_case.metadata.get("benchmark_type", provenance)),
        forward_model=str(wave_case.metadata.get("forward_model", "wavefield_derived_features")),
        feature_source=selected_source,
        uses_complex_wavefield=freq_data is not None,
        simulation_qc_passed=wave_case.metadata.get("simulation_qc_passed"),
        extra={
            "measurement_domain": "features",
            "feature_extraction": {
                "method": method,
                "selected_delta_tof": selected_source,
                "single_frequency_phase_allowed": False,
                "water_reference_handling": "required",
            },
            "feature_qc": qc,
        },
    )
    feature_case = wave_case.model_copy(
        update={
            "case_id": f"{wave_case.case_id}_features",
            "measurement": MeasurementSpec(
                domain="features",
                frequencies_hz=frequencies if frequencies.size else None,
                freq_data=freq_data,
                time_data=time_data,
                water_reference=reference,
                source_wavelet=wave_case.measurement.source_wavelet,
                time_axis_s=time_axis,
                tof_s=first_arrival,
                delta_tof_s=selected_delta,
                tof_first_arrival_s=first_delta,
                tof_xcorr_s=xcorr_delay,
                phase_slope_delay_s=phase_delay,
                log_amp=log_amp_for_solver,
                valid_mask=selected_valid,
                feature_quality=feature_quality,
            ),
            "metadata": metadata,
        }
    )
    if out is not None:
        out_path = Path(out)
        write_case_hdf5(feature_case, out_path)
        _write_feature_artifacts(feature_case, qc, out_path.parent)
    return feature_case, qc


def _non_self_mask(case: USCTCase) -> np.ndarray:
    shape = (case.geometry.tx_pos_m.shape[0], case.geometry.rx_pos_m.shape[0])
    mask = np.ones(shape, dtype=bool)
    if shape[0] == shape[1]:
        mask &= ~np.eye(shape[0], dtype=bool)
    return mask


def _feature_quality_map(
    *,
    first_delta: np.ndarray,
    xcorr_delay: np.ndarray,
    phase_delay: np.ndarray,
    phase_valid: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    quality = np.zeros_like(xcorr_delay, dtype=float)
    quality[valid_mask] += 0.5
    finite_first = np.isfinite(first_delta) & np.isfinite(xcorr_delay)
    agreement = finite_first & (np.abs(first_delta - xcorr_delay) <= 2.5e-7)
    quality[agreement] += 0.25
    finite_phase = np.isfinite(phase_delay) & np.isfinite(xcorr_delay) & phase_valid
    phase_agreement = finite_phase & (np.abs(phase_delay - xcorr_delay) <= 2.5e-7)
    quality[phase_agreement] += 0.25
    return np.clip(quality, 0.0, 1.0)


def _feature_qc(
    *,
    first_delta: np.ndarray,
    xcorr_delay: np.ndarray,
    phase_delay: np.ndarray,
    phase_valid: np.ndarray,
    valid_mask: np.ndarray,
    log_amp: np.ndarray,
) -> dict[str, Any]:
    valid_fraction = float(np.mean(valid_mask)) if valid_mask.size else 0.0
    first_xcorr_rmse = _rmse(first_delta, xcorr_delay, valid_mask)
    phase_xcorr_rmse = _rmse(phase_delay, xcorr_delay, valid_mask & phase_valid)
    phase_valid_fraction = float(np.mean(phase_valid)) if phase_valid.size else 0.0
    finite_log = np.asarray(log_amp, dtype=float)
    finite_log = finite_log[np.isfinite(finite_log)]
    dynamic_range = float(np.nanmax(finite_log) - np.nanmin(finite_log)) if finite_log.size else 0.0
    return {
        "tof_valid_fraction": valid_fraction,
        "phase_slope_valid_fraction": phase_valid_fraction,
        "first_arrival_vs_xcorr_rmse_s": first_xcorr_rmse,
        "phase_slope_vs_xcorr_rmse_s": phase_xcorr_rmse,
        "log_amp_dynamic_range": dynamic_range,
        "phase_unwrap_failure_fraction": float(1.0 - phase_valid_fraction),
        "bad_tx_rx_fraction": float(1.0 - valid_fraction),
    }


def _rmse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    finite = np.asarray(mask, dtype=bool) & np.isfinite(a) & np.isfinite(b)
    if not np.any(finite):
        return float("nan")
    diff = np.asarray(a, dtype=float)[finite] - np.asarray(b, dtype=float)[finite]
    return float(np.sqrt(np.mean(diff**2)))


def _write_feature_artifacts(case: USCTCase, qc: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if case.measurement.delta_tof_s is not None:
        write_preview_png(case.measurement.delta_tof_s, out_dir / "tof_sinogram_preview.png")
    if case.measurement.log_amp is not None:
        write_preview_png(case.measurement.log_amp, out_dir / "log_amplitude_preview.png")
    if case.measurement.feature_quality is not None:
        write_preview_png(case.measurement.feature_quality, out_dir / "feature_preview.png")
    (out_dir / "feature_quality.json").write_text(json.dumps(qc, indent=2, sort_keys=True), encoding="utf-8")
