"""k-Wave-compatible forward simulation entry points.

The default backend is a deterministic smoke-size wavefield generator used to
validate cache/QC/feature plumbing without requiring CUDA or a full k-Wave
install on the local Mac. Formal high-fidelity runs should set metadata/backend
to an actual k-Wave execution path on A100.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from usctbench.algorithms.ray.straight_projector import StraightRayProjector
from usctbench.data.synthetic import circular_attenuation, make_sound_speed_case
from usctbench.features.phase_delay import frequency_response_from_time
from usctbench.io.hdf5 import read_case_hdf5, write_case_hdf5
from usctbench.provenance import MeasurementProvenance, stamp_measurement_metadata
from usctbench.schema import GeometrySpec, GridSpec, GroundTruthSpec, MeasurementSpec, USCTCase
from usctbench.sim.cache import _array_digest, cached_case_matches, kwave_cache_key

_ENV_DEFAULT_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]*)\}")
_NATIVE_SMOKE_BACKENDS = {"native_smoke", "smoke", "analytic_smoke"}
_EXTERNAL_KWAVE_BACKENDS = {"kwave_a100", "external_kwave", "real_kwave"}


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
    wave_case = simulate_kwave_forward(
        property_case,
        simulation=simulation,
        config_name=str(config.get("name", config_file.stem)),
        cache_key=cache_key,
        outputs=dict(config.get("outputs", {})),
    )
    write_case_hdf5(wave_case, output_path)
    return output_path


def simulate_kwave_forward(
    property_case: USCTCase,
    *,
    simulation: dict[str, Any],
    config_name: str = "kwave_simulation",
    cache_key: str | None = None,
    outputs: dict[str, Any] | None = None,
) -> USCTCase:
    """Generate a wavefield case from a property-map case."""

    backend = str(simulation.get("backend", "native_smoke")).lower()
    if backend in _NATIVE_SMOKE_BACKENDS:
        return _simulate_native_smoke_forward(
            property_case,
            simulation=simulation,
            config_name=config_name,
            cache_key=cache_key,
        )
    if backend in _EXTERNAL_KWAVE_BACKENDS:
        return _simulate_external_kwave_forward(
            property_case,
            simulation=simulation,
            config_name=config_name,
            cache_key=cache_key,
            outputs=outputs or {},
        )
    raise ValueError(f"unsupported k-Wave simulation backend: {backend}")


def _simulate_native_smoke_forward(
    property_case: USCTCase,
    *,
    simulation: dict[str, Any],
    config_name: str,
    cache_key: str | None,
) -> USCTCase:
    """Generate a deterministic smoke-size wavefield for plumbing/QC only."""

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


def _simulate_external_kwave_forward(
    property_case: USCTCase,
    *,
    simulation: dict[str, Any],
    config_name: str,
    cache_key: str | None,
    outputs: dict[str, Any],
) -> USCTCase:
    """Run the A100 external k-Wave forward-only backend and standardize output."""

    if property_case.ground_truth.sound_speed_mps is None:
        raise ValueError("external k-Wave forward requires ground_truth.sound_speed_mps")
    c0 = float(simulation.get("reference_sound_speed_mps", property_case.metadata.get("reference_sound_speed_mps", 1500.0)))
    root = Path(_expand(str(outputs.get("external_root", outputs.get("root", "runs/usctbench_runs/kwave_real_forward"))))).resolve()
    root.mkdir(parents=True, exist_ok=True)
    mat_key = str(simulation.get("mat_key", "usct_property"))
    property_mat = Path(_expand(str(outputs.get("property_mat", root / "property_case.mat")))).resolve()
    water_mat = Path(_expand(str(outputs.get("water_mat", root / "water_case.mat")))).resolve()
    object_dataset = Path(_expand(str(outputs.get("object_dataset", root / "object_dataset.mat")))).resolve()
    water_dataset = Path(_expand(str(outputs.get("water_dataset", root / "water_dataset.mat")))).resolve()
    object_siminfo = Path(_expand(str(outputs.get("object_siminfo", root / "object_siminfo.mat")))).resolve()
    water_siminfo = Path(_expand(str(outputs.get("water_siminfo", root / "water_siminfo.mat")))).resolve()
    object_summary = Path(_expand(str(outputs.get("object_summary", root / "object_forward_summary.json")))).resolve()
    water_summary = Path(_expand(str(outputs.get("water_summary", root / "water_forward_summary.json")))).resolve()

    _write_property_mat(property_case.ground_truth.sound_speed_mps, property_mat, key=mat_key)
    water_speed = np.full(property_case.grid.shape, c0, dtype=float)
    _write_property_mat(water_speed, water_mat, key=mat_key)

    _run_external_forward(
        simulation=simulation,
        mat_path=property_mat,
        mat_key=mat_key,
        siminfo_path=object_siminfo,
        dataset_path=object_dataset,
        summary_path=object_summary,
        water_reference=False,
    )
    _run_external_forward(
        simulation=simulation,
        mat_path=water_mat,
        mat_key=mat_key,
        siminfo_path=water_siminfo,
        dataset_path=water_dataset,
        summary_path=water_summary,
        water_reference=True,
    )
    return _external_datasets_to_case(
        property_case,
        object_dataset=object_dataset,
        water_dataset=water_dataset,
        object_siminfo=object_siminfo,
        simulation=simulation,
        config_name=config_name,
        cache_key=cache_key,
        paths={
            "external_root": root,
            "property_mat": property_mat,
            "water_mat": water_mat,
            "object_dataset": object_dataset,
            "water_dataset": water_dataset,
            "object_siminfo": object_siminfo,
            "water_siminfo": water_siminfo,
            "object_summary": object_summary,
            "water_summary": water_summary,
        },
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
        inclusion_radius_m=float(case_config.get("inclusion_radius_m", 0.006)),
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
        "forward_map_identity": _forward_map_identity_metadata(
            property_case,
            axis_conversion="none_internal_row_y_col_x",
            crop_resample="none",
        ),
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
                "native_smoke validates the unified data path and is not a replacement for high-fidelity A100 k-Wave runs",
            ],
        },
    )


def _external_datasets_to_case(
    property_case: USCTCase,
    *,
    object_dataset: Path,
    water_dataset: Path,
    object_siminfo: Path,
    simulation: dict[str, Any],
    config_name: str,
    cache_key: str | None,
    paths: dict[str, Path],
) -> USCTCase:
    object_data = _read_external_kwave_dataset(object_dataset)
    water_data = _read_external_kwave_dataset(water_dataset)
    time_axis = np.asarray(object_data["time_axis_s"], dtype=float)
    time_data = np.asarray(object_data["time_data"], dtype=float)
    water_reference = _resample_time_data_to_axis(
        np.asarray(water_data["time_data"], dtype=float),
        np.asarray(water_data["time_axis_s"], dtype=float),
        time_axis,
    )
    if time_data.shape != water_reference.shape:
        raise ValueError(f"object/water k-Wave dataset shape mismatch after resampling: {time_data.shape} vs {water_reference.shape}")
    frequencies_hz = np.asarray(simulation.get("frequencies_hz", [150_000.0, 250_000.0, 350_000.0, 450_000.0]), dtype=float)
    freq_data = frequency_response_from_time(time_data, time_axis, frequencies_hz)
    source_wavelet = _read_source_wavelet(object_siminfo)
    sound_speed = np.asarray(object_data["sound_speed_mps"], dtype=float)
    attenuation = np.asarray(object_data["attenuation_np_per_m"], dtype=float)
    density = np.full(sound_speed.shape, float(simulation.get("density_kg_per_m3", 1000.0)), dtype=float)
    geometry = object_data["geometry"]
    grid = object_data["grid"]
    valid = np.ones(time_data.shape[:2], dtype=bool)
    if valid.shape[0] == valid.shape[1]:
        valid &= ~np.eye(valid.shape[0], dtype=bool)
    pml = int(simulation.get("pml_thickness_pixels", simulation.get("pml_size", object_data["metadata"].get("PMLSize", 0))))
    cfl = float(simulation.get("cfl_number", simulation.get("cfl", object_data["metadata"].get("cfl", 0.0))))
    peak_frequency = float(simulation.get("source_peak_frequency_hz", simulation.get("f_tx_hz", 250_000.0)))
    source_bandwidth = float(simulation.get("source_bandwidth_hz", simulation.get("frac_bw", 0.75) * peak_frequency))
    backend = str(simulation.get("backend", "kwave_a100"))
    actual_backend = str(simulation.get("external_backend", simulation.get("kwave_backend", "cuda-binary")))
    sim_meta = {
        "config_name": config_name,
        "backend": backend,
        "actual_kwave_backend": actual_backend,
        "cache_key": cache_key,
        "grid_shape": list(grid.shape),
        "spacing_m": list(grid.spacing_m),
        "n_tx": int(time_data.shape[0]),
        "n_rx": int(time_data.shape[1]),
        "reference_sound_speed_mps": float(simulation.get("reference_sound_speed_mps", 1500.0)),
        "source_wavelet": simulation.get("source_wavelet", "gauspuls"),
        "source_peak_frequency_hz": peak_frequency,
        "source_bandwidth_hz": source_bandwidth,
        "dt_s": float(np.median(np.diff(time_axis))) if time_axis.size > 1 else None,
        "n_time": int(time_axis.size),
        "pml_thickness_pixels": pml,
        "cfl_number": cfl,
        "frequencies_hz": [float(value) for value in frequencies_hz.tolist()],
        "external_paths": {key: str(path) for key, path in paths.items()},
        "dataset_metadata": object_data["metadata"],
        "water_time_axis_original_length": int(np.asarray(water_data["time_axis_s"]).size),
        "object_time_axis_length": int(time_axis.size),
        "forward_map_identity": _forward_map_identity_metadata(
            property_case,
            sound_speed=sound_speed,
            attenuation=attenuation,
            density=density,
            grid=grid,
            axis_conversion="external_kwave_matlab_xy_transposed_to_internal_row_y_col_x",
            crop_resample=str(object_data["metadata"].get("array_axis_conversion", "transpose_external_xy_to_internal_yx")),
        ),
    }
    metadata = stamp_measurement_metadata(
        property_case.metadata,
        measurement_provenance=MeasurementProvenance.SELF_SIMULATED_KWAVE_WAVEFIELD,
        benchmark_type="self_simulated_kwave_wavefield",
        forward_model="external_k_wave_forward_only",
        feature_source="raw_time_and_frequency_wavefield",
        uses_complex_wavefield=True,
        extra={
            "case_type": "kwave_wavefield",
            "measurement_domain": "time",
            "simulation_backend": backend,
            "external_kwave_backend": actual_backend,
            "simulation_cache_key": cache_key,
            "simulation_metadata": sim_meta,
            "measurement_limitations": [
                "real external k-Wave forward-only self-simulation; requires QC before algorithm conclusions",
                "water_reference was generated by a separate homogeneous-water k-Wave forward run",
                "ground-truth property map generated the measurement, so inverse-crime risk remains medium",
            ],
        },
    )
    return USCTCase(
        case_id=f"{property_case.case_id}_kwave_wavefield",
        grid=grid,
        geometry=geometry,
        measurement=MeasurementSpec(
            domain="time",
            frequencies_hz=frequencies_hz,
            freq_data=freq_data,
            time_data=time_data,
            water_reference=water_reference,
            source_wavelet=source_wavelet,
            time_axis_s=time_axis,
            valid_mask=valid,
        ),
        ground_truth=GroundTruthSpec(
            sound_speed_mps=sound_speed,
            attenuation_np_per_m=attenuation,
            density_kg_per_m3=density,
        ),
        metadata=metadata,
    )


def _run_external_forward(
    *,
    simulation: dict[str, Any],
    mat_path: Path,
    mat_key: str,
    siminfo_path: Path,
    dataset_path: Path,
    summary_path: Path,
    water_reference: bool,
) -> None:
    if dataset_path.exists() and not bool(simulation.get("overwrite", False)):
        return
    runner = Path(__file__).with_name("external_kwave_runner.py")
    python_bin = str(simulation.get("python_bin") or os.environ.get("USCT_KWAVE_PYTHON_BIN") or sys.executable)
    usct_kwave_root = str(simulation.get("usct_kwave_root") or os.environ.get("USCT_KWAVE_ROOT") or Path.home() / "USCT_kwave")
    command = [
        python_bin,
        str(runner),
        "--usct-kwave-root",
        usct_kwave_root,
        "--mat-path",
        str(mat_path),
        "--mat-key",
        mat_key,
        "--sample-index",
        "1",
        "--array-mode",
        str(simulation.get("array_mode", "partial64")),
        "--background-speed",
        str(simulation.get("reference_sound_speed_mps", 1500.0)),
        "--ncalc",
        str(simulation.get("ncalc", 552)),
        "--xmax-mm",
        str(simulation.get("xmax_mm", 120.0)),
        "--circle-radius-mm",
        str(simulation.get("circle_radius_mm", 110.0)),
        "--atten-bkgnd",
        str(0.0 if water_reference else simulation.get("atten_bkgnd", 0.0)),
        "--sos2atten",
        str(0.0 if water_reference else simulation.get("sos2atten", 0.0)),
        "--y-atten",
        str(simulation.get("y_atten", 1.01)),
        "--f-tx-mhz",
        str(float(simulation.get("source_peak_frequency_hz", 250_000.0)) / 1.0e6),
        "--frac-bw",
        str(simulation.get("frac_bw", 0.75)),
        "--cfl",
        str(simulation.get("cfl_number", simulation.get("cfl", 0.3))),
        "--pml-size",
        str(simulation.get("pml_thickness_pixels", simulation.get("pml_size", 20))),
        "--downsample-factor",
        str(simulation.get("downsample_factor", 1)),
        "--backend",
        str(simulation.get("external_backend", simulation.get("kwave_backend", "cuda-binary"))),
        "--binary-path",
        str(simulation.get("binary_path", "")),
        "--generation-mode",
        str(simulation.get("generation_mode", "direct")),
        "--direct-num-workers",
        str(simulation.get("direct_num_workers", 1)),
        "--kwave-data-path",
        str(simulation.get("kwave_data_path", "auto")),
        "--kwave-data-name-prefix",
        str(simulation.get("kwave_data_name_prefix", dataset_path.stem)),
        "--siminfo-path",
        str(siminfo_path),
        "--dataset-path",
        str(dataset_path),
        "--summary-path",
        str(summary_path),
    ]
    for device in simulation.get("cuda_devices", []):
        command.extend(["--cuda-device", str(device)])
    if bool(simulation.get("overwrite", False)):
        command.append("--overwrite")
    if bool(simulation.get("start_matlab", True)):
        command.append("--start-matlab")
    if bool(simulation.get("no_connect_existing", True)):
        command.append("--no-connect-existing")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{usct_kwave_root}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    timeout_s = int(simulation.get("timeout_s", 7200))
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, timeout=timeout_s, check=False)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = summary_path.with_suffix(".log")
    log_path.write_text(f"$ {' '.join(command)}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}\n", encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"external k-Wave forward failed with exit {proc.returncode}; see {log_path}")


def _write_property_mat(sound_speed_mps: np.ndarray, path: Path, *, key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from scipy.io import savemat
    except ModuleNotFoundError as exc:
        raise RuntimeError("scipy is required to write MATLAB property maps for external k-Wave") from exc
    savemat(path, {key: np.asarray(sound_speed_mps, dtype=np.float32)}, do_compression=False)


def _read_external_kwave_dataset(path: Path) -> dict[str, Any]:
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError("h5py is required to read external k-Wave datasets") from exc
    with h5py.File(path, "r") as handle:
        time_axis = np.asarray(handle["time"][()], dtype=float).reshape(-1)
        positions_xy = np.asarray(handle["transducerPositionsXY"][()], dtype=float)
        tx_indices = _read_indices(handle.get("tx_indices_saved"), positions_xy.shape[0])
        rx_indices = _read_indices(handle.get("rx_indices_saved"), positions_xy.shape[0])
        full_dataset = _as_tx_rx_time(handle["full_dataset"][()], time_axis.size, rx_indices.size, tx_indices.size)
        sound_speed_raw = np.asarray(handle["C"][()], dtype=float)
        attenuation_raw = np.asarray(handle["atten"][()], dtype=float)
        sound_speed = _external_xy_image_to_internal_yx(sound_speed_raw)
        attenuation = _external_xy_image_to_internal_yx(attenuation_raw)
        xi = np.asarray(handle["xi_orig"][()], dtype=float).reshape(-1) if "xi_orig" in handle else None
        yi = np.asarray(handle["yi_orig"][()], dtype=float).reshape(-1) if "yi_orig" in handle else None
        metadata = _read_hdf5_group(handle["sim_metadata"]) if "sim_metadata" in handle else {}
    grid = _grid_from_coordinates(sound_speed.shape, xi=xi, yi=yi)
    geometry = _geometry_from_xy_positions(positions_xy, tx_indices=tx_indices, rx_indices=rx_indices)
    return {
        "time_axis_s": time_axis,
        "time_data": full_dataset,
        "sound_speed_mps": sound_speed,
        "attenuation_np_per_m": attenuation,
        "grid": grid,
        "geometry": geometry,
        "metadata": {
            **metadata,
            "array_axis_convention_raw": "[x,y]",
            "array_axis_convention_internal": "[row=y,col=x]",
            "array_axis_conversion": "transpose_external_xy_to_internal_yx",
            "raw_sound_speed_shape_xy": [int(value) for value in sound_speed_raw.shape],
            "internal_sound_speed_shape_yx": [int(value) for value in sound_speed.shape],
        },
    }


def _forward_map_identity_metadata(
    case: USCTCase,
    *,
    sound_speed: np.ndarray | None = None,
    attenuation: np.ndarray | None = None,
    density: np.ndarray | None = None,
    grid: GridSpec | None = None,
    axis_conversion: str,
    crop_resample: str,
) -> dict[str, Any]:
    """Describe the property maps used by forward simulation and metric GT.

    USCTCase image arrays are always stored internally as `[row=y, col=x]`.
    This metadata lets downstream reports verify that the map used by the
    forward model is the same map later used as metric ground truth.
    """

    sim_grid = grid or case.grid
    speed = case.ground_truth.sound_speed_mps if sound_speed is None else sound_speed
    atten = case.ground_truth.attenuation_np_per_m if attenuation is None else attenuation
    dens = case.ground_truth.density_kg_per_m3 if density is None else density
    roi = sim_grid.roi_mask
    return {
        "source_case_id": case.case_id,
        "array_axis_convention_internal": "[row=y,col=x]",
        "external_axis_conversion": axis_conversion,
        "crop_resample_metadata": crop_resample,
        "grid_shape": [int(value) for value in sim_grid.shape],
        "spacing_m": [float(value) for value in sim_grid.spacing_m],
        "origin_m": [float(value) for value in sim_grid.origin_m],
        "roi_hash": _array_digest(roi),
        "geometry_hash": _geometry_digest(case.geometry),
        "forward_sound_speed_hash": _array_digest(speed),
        "forward_sound_speed_shape": _array_shape(speed),
        "forward_density_hash": _array_digest(dens),
        "forward_density_shape": _array_shape(dens),
        "forward_attenuation_hash": _array_digest(atten),
        "forward_attenuation_shape": _array_shape(atten),
        "metric_gt_sound_speed_hash": _array_digest(speed),
        "metric_gt_density_hash": _array_digest(dens),
        "metric_gt_attenuation_hash": _array_digest(atten),
        "metric_gt_role": "same_internal_arrays_stored_as_ground_truth_for_metric_evaluation",
        "forward_map_vs_metric_gt_expected_rmse": 0.0,
    }


def _array_shape(value: np.ndarray | None) -> list[int] | None:
    if value is None:
        return None
    return [int(size) for size in np.asarray(value).shape]


def _geometry_digest(geometry: GeometrySpec) -> str:
    payload = np.concatenate(
        [
            np.ascontiguousarray(np.asarray(geometry.tx_pos_m, dtype=float)).reshape(-1),
            np.ascontiguousarray(np.asarray(geometry.rx_pos_m, dtype=float)).reshape(-1),
        ]
    )
    return str(_array_digest(payload))


def _read_source_wavelet(siminfo_path: Path) -> np.ndarray:
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError("h5py is required to read k-Wave siminfo") from exc
    with h5py.File(siminfo_path, "r") as handle:
        if "tx_signal" not in handle:
            return np.asarray([], dtype=float)
        return np.asarray(handle["tx_signal"][()], dtype=float).reshape(-1)


def _as_tx_rx_time(data: np.ndarray, n_time: int, n_rx: int, n_tx: int) -> np.ndarray:
    array = np.asarray(data, dtype=np.float32).squeeze()
    if array.shape == (n_time, n_rx, n_tx):
        return np.transpose(array, (2, 1, 0))
    if array.shape == (n_tx, n_rx, n_time):
        return array
    if array.shape == (n_rx, n_tx, n_time):
        return np.transpose(array, (1, 0, 2))
    raise ValueError(f"full_dataset shape {array.shape} does not match n_time={n_time}, n_rx={n_rx}, n_tx={n_tx}")


def _external_xy_image_to_internal_yx(image: np.ndarray) -> np.ndarray:
    """Convert external k-Wave/MATLAB image arrays from [x,y] to [row=y,col=x].

    MATLAB/k-Wave stores the image plane in x-major order for these external
    datasets. USCTCase image-domain arrays are always row-major [y,x].
    """

    array = np.asarray(image, dtype=float).squeeze()
    if array.ndim != 2:
        raise ValueError(f"external k-Wave image arrays must be 2-D [x,y], got {array.shape}")
    return np.ascontiguousarray(array.T)


def _resample_time_data_to_axis(data: np.ndarray, source_axis: np.ndarray, target_axis: np.ndarray) -> np.ndarray:
    source = np.asarray(source_axis, dtype=float).reshape(-1)
    target = np.asarray(target_axis, dtype=float).reshape(-1)
    values = np.asarray(data, dtype=np.float32)
    if source.size == target.size and np.allclose(source, target, rtol=1.0e-6, atol=1.0e-12):
        return values
    if source.size < 2 or target.size < 2:
        raise ValueError("cannot resample k-Wave reference with fewer than two time samples")
    source_dt = float(np.median(np.diff(source)))
    target_dt = float(np.median(np.diff(target)))
    if np.isclose(source_dt, target_dt, rtol=1.0e-4, atol=1.0e-12) and target[0] >= source[0] - 1.0e-12:
        out = np.zeros(values.shape[:2] + (target.size,), dtype=np.float32)
        offset = int(round((source[0] - target[0]) / target_dt))
        start_target = max(0, offset)
        start_source = max(0, -offset)
        count = min(values.shape[-1] - start_source, target.size - start_target)
        if count > 0:
            out[..., start_target : start_target + count] = values[..., start_source : start_source + count]
        return out
    flat = values.reshape((-1, values.shape[-1]))
    out_flat = np.zeros((flat.shape[0], target.size), dtype=np.float32)
    for row, trace in enumerate(flat):
        out_flat[row] = np.interp(target, source, trace, left=0.0, right=0.0).astype(np.float32)
    return out_flat.reshape(values.shape[:2] + (target.size,))


def _read_indices(dataset: Any, default_count: int) -> np.ndarray:
    if dataset is None:
        return np.arange(default_count, dtype=int)
    values = np.asarray(dataset[()], dtype=int).reshape(-1)
    if values.size == 0:
        return np.arange(default_count, dtype=int)
    return values - 1 if int(np.min(values)) >= 1 else values


def _geometry_from_xy_positions(positions_xy: np.ndarray, *, tx_indices: np.ndarray, rx_indices: np.ndarray) -> GeometrySpec:
    positions = np.asarray(positions_xy, dtype=float).squeeze()
    if positions.ndim != 2:
        raise ValueError(f"transducerPositionsXY must be 2-D, got {positions.shape}")
    if positions.shape[0] == 2:
        positions = positions.T
    if positions.shape[1] != 2:
        raise ValueError(f"transducerPositionsXY must have 2 coordinates, got {positions.shape}")
    positions_yx = positions[:, [1, 0]]
    tx_pos = positions_yx[np.asarray(tx_indices, dtype=int)]
    rx_pos = positions_yx[np.asarray(rx_indices, dtype=int)]
    radius_m = float(np.nanmedian(np.linalg.norm(positions_yx, axis=1)))
    return GeometrySpec(type="ring", tx_pos_m=tx_pos, rx_pos_m=rx_pos, radius_m=radius_m)


def _grid_from_coordinates(shape: tuple[int, int], *, xi: np.ndarray | None, yi: np.ndarray | None) -> GridSpec:
    if xi is not None and yi is not None and xi.size > 1 and yi.size > 1:
        x_min, x_max = float(np.nanmin(xi)), float(np.nanmax(xi))
        y_min, y_max = float(np.nanmin(yi)), float(np.nanmax(yi))
        dx = (x_max - x_min) / float(shape[1] - 1 if shape[1] > 1 else shape[1])
        dy = (y_max - y_min) / float(shape[0] - 1 if shape[0] > 1 else shape[0])
        if dy > 0.0 and dx > 0.0:
            return GridSpec(shape=shape, spacing_m=(dy, dx), origin_m=(y_min, x_min), roi_mask=np.ones(shape, dtype=bool))
    return GridSpec(shape=shape, spacing_m=(1.0e-3, 1.0e-3), roi_mask=np.ones(shape, dtype=bool))


def _read_hdf5_group(group: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in group.items():
        if hasattr(value, "keys"):
            out[key] = _read_hdf5_group(value)
        else:
            array = np.asarray(value[()])
            out[key] = _decode_hdf5_value(array)
    return out


def _decode_hdf5_value(array: np.ndarray) -> Any:
    squeezed = np.asarray(array).squeeze()
    if squeezed.dtype.kind in {"U", "S"}:
        return str(squeezed)
    if squeezed.dtype == np.uint16 and squeezed.ndim >= 1:
        chars = [chr(int(value)) for value in squeezed.reshape(-1) if int(value) != 0]
        if chars and all(ch.isprintable() for ch in chars):
            return "".join(chars)
    if squeezed.ndim == 0:
        value = squeezed.item()
        if isinstance(value, np.generic):
            return value.item()
        return value
    return squeezed.tolist()


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
