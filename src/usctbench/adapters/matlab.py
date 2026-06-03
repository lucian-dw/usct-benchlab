"""MATLAB adapter utilities with explicit graceful-skip behavior."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from usctbench.features.phase_delay import frequency_response_from_time
from usctbench.schema import ReconstructionResult, USCTCase


class MatlabUnavailable(RuntimeError):
    """Raised when a MATLAB-backed adapter cannot be executed."""


def find_matlab(configured_bin: str | None = None) -> str | None:
    """Find a MATLAB executable from config, environment, or PATH."""

    candidates = [configured_bin, os.environ.get("MATLAB_BIN"), shutil.which("matlab")]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
        if candidate and shutil.which(candidate):
            return str(shutil.which(candidate))
    return None


@dataclass(frozen=True)
class MatlabAdapter:
    """Small wrapper around `matlab -batch` for optional classic methods."""

    matlab_bin: str
    work_dir: Path

    @classmethod
    def from_config(cls, *, matlab_bin: str | None = None, work_dir: str | Path | None = None) -> "MatlabAdapter":
        resolved = find_matlab(matlab_bin)
        if resolved is None:
            raise MatlabUnavailable("MATLAB executable not found; set MATLAB_BIN or parameters.matlab_bin")
        return cls(matlab_bin=resolved, work_dir=Path(work_dir or ".").resolve())

    def run_batch(self, code: str, *, log_name: str = "matlab.log", timeout_s: int | None = None) -> Path:
        """Run MATLAB batch code and save stdout/stderr to a log file."""

        self.work_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.work_dir / log_name
        completed = subprocess.run(
            [self.matlab_bin, "-batch", code],
            cwd=self.work_dir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            check=False,
        )
        log_path.write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            raise MatlabUnavailable(f"MATLAB command failed with exit code {completed.returncode}; see {log_path}")
        return log_path


def write_usct_case_mat(case: USCTCase, path: str | Path) -> Path:
    """Write a MATLAB-readable HDF5 `.mat` adapter input file.

    The file is intentionally schema-first rather than tailored to one external
    repository. MATLAB entrypoints can read it with `h5read`/`h5readatt`, then
    reshape or rename fields for refraction-corrected or r-Wave code.
    """

    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError("h5py is required to write MATLAB adapter input files") from exc

    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as handle:
        handle.attrs["schema_version"] = "usctbench-matlab-adapter-v0.1"
        handle.attrs["record_type"] = "USCTCase"
        handle.attrs["case_id"] = case.case_id
        handle.attrs["metadata_json"] = _json_dumps(case.metadata)

        grid = handle.create_group("grid")
        grid.create_dataset("shape", data=np.asarray(case.grid.shape, dtype=np.int64))
        grid.create_dataset("spacing_m", data=np.asarray(case.grid.spacing_m, dtype=float))
        grid.create_dataset("origin_m", data=np.asarray(case.grid.origin_m, dtype=float))
        _write_optional_dataset(grid, "roi_mask", _as_numeric_bool(case.grid.roi_mask))

        geometry = handle.create_group("geometry")
        geometry.attrs["type"] = str(case.geometry.type)
        if case.geometry.radius_m is not None:
            geometry.attrs["radius_m"] = float(case.geometry.radius_m)
        geometry.create_dataset("tx_pos_m", data=np.asarray(case.geometry.tx_pos_m, dtype=float))
        geometry.create_dataset("rx_pos_m", data=np.asarray(case.geometry.rx_pos_m, dtype=float))

        measurement = handle.create_group("measurement")
        measurement.attrs["domain"] = str(case.measurement.domain)
        for name in (
            "frequencies_hz",
            "freq_data",
            "time_data",
            "water_reference",
            "source_wavelet",
            "time_axis_s",
            "tof_s",
            "delta_tof_s",
            "tof_first_arrival_s",
            "tof_xcorr_s",
            "phase_slope_delay_s",
            "log_amp",
            "feature_quality",
            "ray_weights",
        ):
            _write_optional_dataset(measurement, name, getattr(case.measurement, name))
        _write_optional_dataset(measurement, "valid_mask", _as_numeric_bool(case.measurement.valid_mask))

        ground_truth = handle.create_group("ground_truth")
        _write_optional_dataset(ground_truth, "sound_speed_mps", case.ground_truth.sound_speed_mps)
        _write_optional_dataset(ground_truth, "attenuation_np_per_m", case.ground_truth.attenuation_np_per_m)
        _write_rwave_complex_group(handle, case)
    return out


def write_matlab_adapter_result(result: ReconstructionResult, path: str | Path) -> Path:
    """Write the standard MATLAB-adapter output format.

    External MATLAB scripts may write the same HDF5 layout directly. This helper
    exists for tests and Python-side adapters that want an exact reference file.
    """

    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError("h5py is required to write MATLAB adapter output files") from exc

    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as handle:
        handle.attrs["schema_version"] = "usctbench-matlab-adapter-v0.1"
        handle.attrs["record_type"] = "ReconstructionResult"
        handle.attrs["algorithm"] = result.algorithm
        handle.attrs["case_id"] = result.case_id
        handle.attrs["runtime_s"] = float(result.runtime_s)
        handle.attrs["status"] = str(result.status)
        if result.failure_reason:
            handle.attrs["failure_reason"] = result.failure_reason
        handle.attrs["metrics_json"] = _json_dumps(result.metrics)
        handle.attrs["artifacts_json"] = _json_dumps(result.artifacts)
        for name in ("sound_speed_mps", "attenuation_np_per_m", "reflectivity", "uncertainty"):
            _write_optional_dataset(handle, name, getattr(result, name))
    return out


def write_matlab_adapter_contract(directory: str | Path) -> Path:
    """Write first-party MATLAB helper functions for external adapters.

    The helper files are copied into the per-run work directory before MATLAB
    starts. External entrypoints can then call `usctbench_read_case` and
    `usctbench_write_result` without vendoring project Python code or
    re-implementing the HDF5 schema.
    """

    out_dir = Path(directory).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "usctbench_read_case.m").write_text(_MATLAB_READ_CASE, encoding="utf-8")
    (out_dir / "usctbench_write_result.m").write_text(_MATLAB_WRITE_RESULT, encoding="utf-8")
    return out_dir


def read_matlab_adapter_result(path: str | Path, *, algorithm: str, case_id: str) -> ReconstructionResult:
    """Read a MATLAB-adapter output MAT/HDF5 file as `ReconstructionResult`."""

    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError("h5py is required to read MATLAB adapter output files") from exc

    with h5py.File(Path(path).expanduser(), "r") as handle:
        return ReconstructionResult(
            algorithm=_read_attr(handle.attrs, "algorithm", algorithm),
            case_id=_read_attr(handle.attrs, "case_id", case_id),
            sound_speed_mps=_read_optional_dataset(handle, "sound_speed_mps"),
            attenuation_np_per_m=_read_optional_dataset(handle, "attenuation_np_per_m"),
            reflectivity=_read_optional_dataset(handle, "reflectivity"),
            uncertainty=_read_optional_dataset(handle, "uncertainty"),
            metrics=_json_loads(handle.attrs.get("metrics_json", "{}")),
            runtime_s=float(handle.attrs.get("runtime_s", 0.0)),
            status=_read_attr(handle.attrs, "status", "success"),
            failure_reason=_read_attr(handle.attrs, "failure_reason", None),
            artifacts=_json_loads(handle.attrs.get("artifacts_json", "{}")),
        )


def _write_optional_dataset(group: Any, name: str, value: Any) -> None:
    if value is not None:
        group.create_dataset(name, data=np.asarray(value))


def _read_optional_dataset(group: Any, name: str) -> np.ndarray | None:
    if name not in group:
        return None
    return np.asarray(group[name][()])


def _as_numeric_bool(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=np.uint8)


def _write_rwave_complex_group(handle: Any, case: USCTCase) -> None:
    """Write only the retired rWave complex group when explicitly available.

    The current v0.1 mainline keeps rWave on travel-time surrogate inputs and
    reserves k-Wave raw/complex data for FWI.  This adapter therefore no longer
    derives complex rWave features as a side effect of MATLAB export.
    """

    if (
        case.measurement.freq_data is None
        or case.measurement.frequencies_hz is None
        or case.measurement.water_reference is None
        or bool(case.metadata.get("write_retired_rwave_complex_group", False)) is False
    ):
        return
    group = handle.create_group("rwave")
    freq_data = np.asarray(case.measurement.freq_data)
    reference = _reference_frequency_data_for_matlab(case)
    scattered = freq_data - reference
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = freq_data / np.where(np.abs(reference) > 0.0, reference, np.nan)
    group.attrs["scattered_field_convention"] = "scattered_freq_data = freq_data - water_reference_freq_data"
    group.attrs["complex_ratio_convention"] = "complex_ratio = freq_data / water_reference_freq_data"
    group.attrs["retired"] = True
    group.attrs["note"] = "complex rWave export is retired from the main benchmark; k-Wave data is reserved for FWI"
    group.create_dataset("frequencies_hz", data=np.asarray(case.measurement.frequencies_hz, dtype=float))
    group.create_dataset("freq_data_real", data=np.real(freq_data))
    group.create_dataset("freq_data_imag", data=np.imag(freq_data))
    group.create_dataset("reference_freq_data_real", data=np.real(reference))
    group.create_dataset("reference_freq_data_imag", data=np.imag(reference))
    group.create_dataset("scattered_freq_data_real", data=np.real(scattered))
    group.create_dataset("scattered_freq_data_imag", data=np.imag(scattered))
    group.create_dataset("complex_ratio_real", data=np.real(ratio))
    group.create_dataset("complex_ratio_imag", data=np.imag(ratio))


def _reference_frequency_data_for_matlab(case: USCTCase) -> np.ndarray:
    freq_data = np.asarray(case.measurement.freq_data)
    reference = np.asarray(case.measurement.water_reference)
    if reference.shape == freq_data.shape:
        return reference
    if reference.ndim == 3 and case.measurement.time_axis_s is not None:
        return frequency_response_from_time(
            reference.astype(float, copy=False),
            np.asarray(case.measurement.time_axis_s, dtype=float),
            np.asarray(case.measurement.frequencies_hz, dtype=float),
        )
    raise ValueError("water_reference must be frequency-domain or time-domain with time_axis_s")


def _json_dumps(value: dict[str, Any]) -> str:
    return json.dumps(value, default=_json_default, sort_keys=True)


def _json_loads(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(str(value))


def _read_attr(attrs: Any, key: str, default: Any = None) -> Any:
    value = attrs.get(key, default)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


_MATLAB_READ_CASE = """function case_data = usctbench_read_case(input_mat)
case_data = struct();
case_data.case_id = h5readatt(input_mat, '/', 'case_id');
case_data.metadata_json = h5readatt(input_mat, '/', 'metadata_json');
case_data.grid = struct();
case_data.grid.shape = double(h5read(input_mat, '/grid/shape'))';
case_data.grid.spacing_m = double(h5read(input_mat, '/grid/spacing_m'))';
case_data.grid.origin_m = double(h5read(input_mat, '/grid/origin_m'))';
case_data.grid.roi_mask = usctbench_optional_h5read(input_mat, '/grid/roi_mask');
case_data.geometry = struct();
case_data.geometry.type = h5readatt(input_mat, '/geometry', 'type');
case_data.geometry.radius_m = usctbench_optional_h5readatt(input_mat, '/geometry', 'radius_m');
case_data.geometry.tx_pos_m = double(h5read(input_mat, '/geometry/tx_pos_m'));
case_data.geometry.rx_pos_m = double(h5read(input_mat, '/geometry/rx_pos_m'));
case_data.measurement = struct();
case_data.measurement.domain = h5readatt(input_mat, '/measurement', 'domain');
case_data.measurement.delta_tof_s = usctbench_optional_h5read(input_mat, '/measurement/delta_tof_s');
case_data.measurement.tof_s = usctbench_optional_h5read(input_mat, '/measurement/tof_s');
case_data.measurement.valid_mask = usctbench_optional_h5read(input_mat, '/measurement/valid_mask');
case_data.measurement.log_amp = usctbench_optional_h5read(input_mat, '/measurement/log_amp');
case_data.measurement.frequencies_hz = usctbench_optional_h5read(input_mat, '/measurement/frequencies_hz');
case_data.measurement.freq_data = usctbench_optional_h5read(input_mat, '/measurement/freq_data');
case_data.measurement.time_data = usctbench_optional_h5read(input_mat, '/measurement/time_data');
case_data.measurement.water_reference = usctbench_optional_h5read(input_mat, '/measurement/water_reference');
case_data.measurement.source_wavelet = usctbench_optional_h5read(input_mat, '/measurement/source_wavelet');
case_data.measurement.time_axis_s = usctbench_optional_h5read(input_mat, '/measurement/time_axis_s');
case_data.measurement.tof_first_arrival_s = usctbench_optional_h5read(input_mat, '/measurement/tof_first_arrival_s');
case_data.measurement.tof_xcorr_s = usctbench_optional_h5read(input_mat, '/measurement/tof_xcorr_s');
case_data.measurement.phase_slope_delay_s = usctbench_optional_h5read(input_mat, '/measurement/phase_slope_delay_s');
case_data.measurement.feature_quality = usctbench_optional_h5read(input_mat, '/measurement/feature_quality');
case_data.measurement.ray_weights = usctbench_optional_h5read(input_mat, '/measurement/ray_weights');
case_data.rwave = struct();
case_data.rwave.frequencies_hz = usctbench_optional_h5read(input_mat, '/rwave/frequencies_hz');
case_data.rwave.freq_data_real = usctbench_optional_h5read(input_mat, '/rwave/freq_data_real');
case_data.rwave.freq_data_imag = usctbench_optional_h5read(input_mat, '/rwave/freq_data_imag');
case_data.rwave.reference_freq_data_real = usctbench_optional_h5read(input_mat, '/rwave/reference_freq_data_real');
case_data.rwave.reference_freq_data_imag = usctbench_optional_h5read(input_mat, '/rwave/reference_freq_data_imag');
case_data.rwave.scattered_freq_data_real = usctbench_optional_h5read(input_mat, '/rwave/scattered_freq_data_real');
case_data.rwave.scattered_freq_data_imag = usctbench_optional_h5read(input_mat, '/rwave/scattered_freq_data_imag');
case_data.rwave.complex_ratio_real = usctbench_optional_h5read(input_mat, '/rwave/complex_ratio_real');
case_data.rwave.complex_ratio_imag = usctbench_optional_h5read(input_mat, '/rwave/complex_ratio_imag');
case_data.rwave.rytov_data_real = usctbench_optional_h5read(input_mat, '/rwave/rytov_data_real');
case_data.rwave.rytov_data_imag = usctbench_optional_h5read(input_mat, '/rwave/rytov_data_imag');
case_data.rwave.log_amplitude_ratio = usctbench_optional_h5read(input_mat, '/rwave/log_amplitude_ratio');
case_data.rwave.phase_slope_delay_s = usctbench_optional_h5read(input_mat, '/rwave/phase_slope_delay_s');
case_data.rwave.phase_fit_rms_rad = usctbench_optional_h5read(input_mat, '/rwave/phase_fit_rms_rad');
case_data.rwave.complex_valid_mask = usctbench_optional_h5read(input_mat, '/rwave/complex_valid_mask');
case_data.rwave.complex_quality = usctbench_optional_h5read(input_mat, '/rwave/complex_quality');
case_data.ground_truth = struct();
case_data.ground_truth.sound_speed_mps = usctbench_optional_h5read(input_mat, '/ground_truth/sound_speed_mps');
case_data.ground_truth.attenuation_np_per_m = usctbench_optional_h5read(input_mat, '/ground_truth/attenuation_np_per_m');
end

function value = usctbench_optional_h5read(input_mat, dataset_name)
try
    value = h5read(input_mat, dataset_name);
catch
    value = [];
end
end

function value = usctbench_optional_h5readatt(input_mat, object_name, attr_name)
try
    value = h5readatt(input_mat, object_name, attr_name);
catch
    value = [];
end
end
"""


_MATLAB_WRITE_RESULT = """function usctbench_write_result(output_mat, algorithm, case_id, sound_speed_mps, metrics_json, varargin)
if nargin < 5 || isempty(metrics_json)
    metrics_json = '{}';
end
if exist(output_mat, 'file')
    delete(output_mat);
end
h5create(output_mat, '/sound_speed_mps', size(sound_speed_mps), 'Datatype', 'double');
h5write(output_mat, '/sound_speed_mps', double(sound_speed_mps));
h5writeatt(output_mat, '/', 'schema_version', 'usctbench-matlab-adapter-v0.1');
h5writeatt(output_mat, '/', 'record_type', 'ReconstructionResult');
h5writeatt(output_mat, '/', 'algorithm', char(algorithm));
h5writeatt(output_mat, '/', 'case_id', char(case_id));
h5writeatt(output_mat, '/', 'runtime_s', 0.0);
h5writeatt(output_mat, '/', 'status', 'success');
h5writeatt(output_mat, '/', 'metrics_json', char(metrics_json));
h5writeatt(output_mat, '/', 'artifacts_json', '{}');
for idx = 1:2:numel(varargin)
    key = varargin{idx};
    value = varargin{idx + 1};
    if strcmp(key, 'attenuation_np_per_m') && ~isempty(value)
        h5create(output_mat, '/attenuation_np_per_m', size(value), 'Datatype', 'double');
        h5write(output_mat, '/attenuation_np_per_m', double(value));
    elseif strcmp(key, 'reflectivity') && ~isempty(value)
        h5create(output_mat, '/reflectivity', size(value), 'Datatype', 'double');
        h5write(output_mat, '/reflectivity', double(value));
    elseif strcmp(key, 'uncertainty') && ~isempty(value)
        h5create(output_mat, '/uncertainty', size(value), 'Datatype', 'double');
        h5write(output_mat, '/uncertainty', double(value));
    end
end
end
"""
