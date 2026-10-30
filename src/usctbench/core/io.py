"""HDF5 serialization for shared USCT schemas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import h5py

from usctbench.core.schema import (
    GeometrySpec,
    GridSpec,
    GroundTruthSpec,
    MeasurementSpec,
    ReconstructionResult,
    USCTCase,
)
from usctbench.data.pressure import pressure_units

SCHEMA_VERSION = "0.1"


# Pressure units are acquisition facts, not inferred from numerical magnitude.
MEASUREMENT_FIELDS = {
    "frequencies_hz": ("frequency", "Hz"),
    "freq_data": ("frequency,tx,rx", "metadata.pressure_contract.data_units"),
    "time_data": ("time,tx,rx", "metadata.pressure_units"),
    "water_reference": ("frequency,tx,rx", "metadata.pressure_contract.data_units"),
    "water_reference_time": ("time,tx,rx", "metadata.pressure_units"),
    "source_spectrum": ("frequency|frequency,tx", "metadata.source_units"),
    "source_wavelet": ("time", "metadata.source_units"),
    "time_axis_s": ("time", "s"),
    "tof_s": ("tx,rx", "s"),
    "delta_tof_s": ("tx,rx", "s"),
    "tof_first_arrival_s": ("tx,rx", "s"),
    "tof_xcorr_s": ("tx,rx", "s"),
    "phase_slope_delay_s": ("tx,rx", "s"),
    "valid_mask": ("tx,rx|frequency,tx,rx", "1"),
    "feature_quality": ("tx,rx", "1"),
    "ray_weights": ("tx,rx", "1"),
}


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


def _json_dumps(value: dict[str, Any]) -> str:
    return json.dumps(value, default=_json_default, sort_keys=True)


def _json_loads(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(value)


def _read_str_attr(attrs: Any, key: str) -> str:
    value = attrs[key]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _write_dataset(group: Any, name: str, value: np.ndarray | None) -> None:
    if value is not None:
        group.create_dataset(name, data=np.asarray(value))


def _read_dataset(group: Any, name: str) -> np.ndarray | None:
    if name not in group:
        return None
    return group[name][()]


def _prepare_path(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def write_case_hdf5(case: USCTCase, path: str | Path) -> Path:
    """Write a USCT case to the standard HDF5 layout."""

    out = _prepare_path(path)
    with h5py.File(out, "w") as handle:
        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.attrs["record_type"] = "USCTCase"
        handle.attrs["case_id"] = case.case_id
        handle.attrs["metadata_json"] = _json_dumps(case.metadata)

        grid = handle.create_group("grid")
        grid.attrs["shape"] = case.grid.shape
        grid.attrs["spacing_m"] = case.grid.spacing_m
        grid.attrs["origin_m"] = case.grid.origin_m
        _write_dataset(grid, "roi_mask", case.grid.roi_mask)

        geometry = handle.create_group("geometry")
        geometry.attrs["type"] = str(case.geometry.type)
        if case.geometry.radius_m is not None:
            geometry.attrs["radius_m"] = case.geometry.radius_m
        _write_dataset(geometry, "tx_pos_m", case.geometry.tx_pos_m)
        _write_dataset(geometry, "rx_pos_m", case.geometry.rx_pos_m)

        measurement = handle.create_group("measurement")
        measurement.attrs["domain"] = str(case.measurement.domain)
        for name, (axes, units) in MEASUREMENT_FIELDS.items():
            value = getattr(case.measurement, name)
            if value is not None:
                dataset = measurement.create_dataset(name, data=value)
                layout = next(
                    (
                        item
                        for item in axes.split("|")
                        if len(item.split(",")) == value.ndim
                    ),
                    None,
                )
                if layout is None:
                    raise ValueError(
                        f"Unsupported rank for persisted {name}: {value.ndim}"
                    )
                dataset.attrs["axes"] = layout
                dataset.attrs["units"] = (
                    pressure_units(case.metadata)
                    if units == "metadata.pressure_contract.data_units"
                    else (
                        case.metadata.get(units.removeprefix("metadata."), "unknown")
                        if units.startswith("metadata.")
                        else units
                    )
                )

        ground_truth = handle.create_group("ground_truth")
        _write_dataset(
            ground_truth, "sound_speed_mps", case.ground_truth.sound_speed_mps
        )
        _write_dataset(
            ground_truth, "density_kg_per_m3", case.ground_truth.density_kg_per_m3
        )
    return out


def read_case_hdf5(path: str | Path) -> USCTCase:
    """Read a USCT case from the standard HDF5 layout."""

    with h5py.File(path, "r") as handle:
        if _read_str_attr(handle.attrs, "schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported USCTCase schema version")
        case_id = _read_str_attr(handle.attrs, "case_id")
        metadata = _json_loads(handle.attrs.get("metadata_json", ""))

        grid_group = handle["grid"]
        grid = GridSpec(
            shape=tuple(int(value) for value in grid_group.attrs["shape"]),
            spacing_m=tuple(float(value) for value in grid_group.attrs["spacing_m"]),
            origin_m=tuple(float(value) for value in grid_group.attrs["origin_m"]),
            roi_mask=_read_dataset(grid_group, "roi_mask"),
        )

        geometry_group = handle["geometry"]
        geometry = GeometrySpec(
            type=_read_str_attr(geometry_group.attrs, "type"),
            tx_pos_m=_read_dataset(geometry_group, "tx_pos_m"),
            rx_pos_m=_read_dataset(geometry_group, "rx_pos_m"),
            radius_m=geometry_group.attrs.get("radius_m"),
        )

        measurement_group = handle["measurement"]
        for name, (axes, units) in MEASUREMENT_FIELDS.items():
            if name not in measurement_group:
                continue
            attrs = measurement_group[name].attrs
            if "axes" in attrs and _read_str_attr(attrs, "axes") not in axes.split("|"):
                raise ValueError(f"Invalid persisted axes for {name}")
            if (
                "units" in attrs
                and not units.startswith("metadata.")
                and attrs["units"] != units
            ):
                raise ValueError(f"Invalid persisted units for {name}")
        measurement = MeasurementSpec(
            domain=_read_str_attr(measurement_group.attrs, "domain"),
            **{
                name: _read_dataset(measurement_group, name)
                for name in MEASUREMENT_FIELDS
            },
        )

        ground_truth_group = handle["ground_truth"]
        ground_truth = GroundTruthSpec(
            sound_speed_mps=_read_dataset(ground_truth_group, "sound_speed_mps"),
            density_kg_per_m3=_read_dataset(ground_truth_group, "density_kg_per_m3"),
        )
    return USCTCase(
        case_id=case_id,
        grid=grid,
        geometry=geometry,
        measurement=measurement,
        ground_truth=ground_truth,
        metadata=metadata,
    )


def write_result_hdf5(result: ReconstructionResult, path: str | Path) -> Path:
    """Write a reconstruction result to HDF5."""

    out = _prepare_path(path)
    with h5py.File(out, "w") as handle:
        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.attrs["record_type"] = "ReconstructionResult"
        handle.attrs["algorithm"] = result.algorithm
        handle.attrs["case_id"] = result.case_id
        for name in ("experiment_id", "variant"):
            value = getattr(result, name)
            if value is not None:
                handle.attrs[name] = value
        handle.attrs["runtime_s"] = result.runtime_s
        handle.attrs["status"] = str(result.status)
        if result.failure_reason is not None:
            handle.attrs["failure_reason"] = result.failure_reason
        handle.attrs["metrics_json"] = _json_dumps(result.metrics)
        handle.attrs["artifacts_json"] = _json_dumps(result.artifacts)

        _write_dataset(handle, "sound_speed_mps", result.sound_speed_mps)
    return out


def read_result_hdf5(path: str | Path) -> ReconstructionResult:
    """Read a reconstruction result from HDF5."""

    with h5py.File(path, "r") as handle:
        if _read_str_attr(handle.attrs, "schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported reconstruction schema version")
        return ReconstructionResult(
            algorithm=_read_str_attr(handle.attrs, "algorithm"),
            case_id=_read_str_attr(handle.attrs, "case_id"),
            **{
                name: _read_str_attr(handle.attrs, name)
                for name in ("experiment_id", "variant")
                if name in handle.attrs
            },
            sound_speed_mps=_read_dataset(handle, "sound_speed_mps"),
            metrics=_json_loads(handle.attrs.get("metrics_json", "")),
            runtime_s=float(handle.attrs["runtime_s"]),
            status=_read_str_attr(handle.attrs, "status"),
            failure_reason=handle.attrs.get("failure_reason"),
            artifacts=_json_loads(handle.attrs.get("artifacts_json", "")),
        )
