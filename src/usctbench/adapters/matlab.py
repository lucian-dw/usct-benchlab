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

from usctbench.schema import USCTCase


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
        for name in ("frequencies_hz", "freq_data", "time_data", "tof_s", "delta_tof_s", "log_amp"):
            _write_optional_dataset(measurement, name, getattr(case.measurement, name))
        _write_optional_dataset(measurement, "valid_mask", _as_numeric_bool(case.measurement.valid_mask))

        ground_truth = handle.create_group("ground_truth")
        _write_optional_dataset(ground_truth, "sound_speed_mps", case.ground_truth.sound_speed_mps)
        _write_optional_dataset(ground_truth, "attenuation_np_per_m", case.ground_truth.attenuation_np_per_m)
    return out


def _write_optional_dataset(group: Any, name: str, value: Any) -> None:
    if value is not None:
        group.create_dataset(name, data=np.asarray(value))


def _as_numeric_bool(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=np.uint8)


def _json_dumps(value: dict[str, Any]) -> str:
    return json.dumps(value, default=_json_default, sort_keys=True)


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
