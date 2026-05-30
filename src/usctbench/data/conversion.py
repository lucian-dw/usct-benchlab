"""Dataset-specific conversion helpers for standard USCTCase files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from usctbench.algorithms.ray.straight_projector import StraightRayProjector
from usctbench.data.synthetic import make_grid, make_ring_geometry
from usctbench.io.hdf5 import write_case_hdf5
from usctbench.schema import GeometrySpec, GridSpec, GroundTruthSpec, MeasurementSpec, USCTCase


def convert_speed_mat_volume(
    mat_path: str | Path,
    out_dir: str | Path,
    *,
    indices: list[int] | None = None,
    dataset_name: str | None = None,
    case_id_prefix: str | None = None,
    output_shape: tuple[int, int] = (64, 64),
    spacing_m: tuple[float, float] = (1.0e-3, 1.0e-3),
    n_transducers: int = 32,
    reference_sound_speed_mps: float = 1500.0,
) -> list[dict[str, Any]]:
    """Convert selected slices of a MATLAB v7.3 speed volume to USCTCase HDF5.

    This converter is intended for speed-only OpenBreastUS mirrors such as a
    `breast_train_speed.mat` file containing `[ny, nx, n_cases]` sound-speed
    maps. Since those files do not contain measured wavefields, the converter
    generates straight-ray surrogate travel-time features from the speed map and
    a zero log-amplitude field so the classical benchmark harness can be smoked
    end-to-end. Metadata records these assumptions explicitly.
    """

    h5py = _h5py()
    source = Path(mat_path).expanduser().resolve()
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    with h5py.File(source, "r") as handle:
        name = dataset_name or _largest_3d_dataset_name(handle)
        dataset = handle[name]
        if dataset.ndim != 3:
            raise ValueError(f"speed dataset must be 3-D [ny, nx, n_cases], got {dataset.shape}")
        selected_indices = indices if indices is not None else [0]
        for index in selected_indices:
            if index < 0 or index >= dataset.shape[2]:
                raise IndexError(f"case index {index} outside dataset shape {dataset.shape}")
            sound_speed = np.asarray(dataset[:, :, index], dtype=float)
            sound_speed = _downsample_mean(sound_speed, output_shape)
            case_id = f"{case_id_prefix or source.stem}_{index:06d}"
            case = _speed_array_to_case(
                sound_speed,
                case_id=case_id,
                source=source,
                dataset_name=name,
                source_index=index,
                source_shape=tuple(int(v) for v in dataset.shape),
                spacing_m=spacing_m,
                n_transducers=n_transducers,
                reference_sound_speed_mps=reference_sound_speed_mps,
            )
            case_path = out_path / f"{case_id}.h5"
            write_case_hdf5(case, case_path)
            records.append(
                {
                    "case_id": case_id,
                    "path": str(case_path),
                    "source_path": str(source),
                    "source_dataset": name,
                    "source_index": index,
                    "shape": list(sound_speed.shape),
                    "conversion": case.metadata["conversion"],
                    "feature_provenance": case.metadata["feature_provenance"],
                    "measurement_limitations": case.metadata["measurement_limitations"],
                    "has_measured_attenuation": False,
                    "attenuation_evidence": "surrogate_zero_log_amp",
                }
            )
    return records


def speed_mat_metadata(mat_path: str | Path) -> dict[str, Any] | None:
    """Return lightweight metadata for a MATLAB v7.3 speed volume if possible."""

    h5py = _h5py()
    path = Path(mat_path)
    try:
        with h5py.File(path, "r") as handle:
            name = _largest_3d_dataset_name(handle)
            dataset = handle[name]
            return {"dataset": name, "shape": list(dataset.shape), "dtype": str(dataset.dtype)}
    except Exception:
        return None


def convert_kwave_channel_mat(
    mat_path: str | Path,
    out_dir: str | Path,
    *,
    case_id_prefix: str | None = None,
    output_shape: tuple[int, int] = (64, 64),
    n_transducers: int = 32,
) -> list[dict[str, Any]]:
    """Convert a compact k-Wave channel MAT file to a standard USCTCase.

    Supported files contain `C`, `atten`, `full_dataset`, `time`, and
    `transducerPositionsXY`. The standard case stores image-domain sound speed
    and attenuation ground truth, surrogate straight-ray delay features from
    `C`, and attenuation line-integral features from the simulated attenuation
    map. The source channel tensor is not copied into the benchmark case.
    """

    h5py = _h5py()
    source = Path(mat_path).expanduser().resolve()
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    with h5py.File(source, "r") as handle:
        metadata = _kwave_channel_metadata_from_handle(handle)
        if metadata is None:
            raise ValueError(f"not a supported k-Wave channel MAT file: {source}")
        sound_speed = np.asarray(handle["C"][()], dtype=float)
        attenuation = np.asarray(handle["atten"][()], dtype=float)
        positions_xy = np.asarray(handle["transducerPositionsXY"][()], dtype=float)
        time_s = np.asarray(handle["time"][()], dtype=float).reshape(-1)
        full_shape = tuple(int(v) for v in handle["full_dataset"].shape)
        source_label = _decode_matlab_chars(handle.get("sim_metadata/sim_label")) or source.stem
        source_npy_path = _decode_matlab_chars(handle.get("sim_metadata/source_npy_path"))
        frequency_hz = _read_scalar(handle.get("sim_metadata/f_tx"))
        xi = np.asarray(handle["xi_orig"][()], dtype=float).reshape(-1) if "xi_orig" in handle else None
        yi = np.asarray(handle["yi_orig"][()], dtype=float).reshape(-1) if "yi_orig" in handle else None

    sound_speed_small = _downsample_mean(sound_speed, output_shape)
    attenuation_small = _downsample_mean(attenuation, output_shape)
    case_id = case_id_prefix or _safe_case_id(source_label)
    case = _kwave_arrays_to_case(
        sound_speed_small,
        attenuation_small,
        positions_xy=positions_xy,
        time_s=time_s,
        source=source,
        source_label=source_label,
        source_shape=tuple(int(v) for v in sound_speed.shape),
        full_dataset_shape=full_shape,
        source_npy_path=source_npy_path,
        frequency_hz=frequency_hz,
        xi=xi,
        yi=yi,
        n_transducers=n_transducers,
    )
    case = case.model_copy(update={"case_id": case_id})
    case_path = out_path / f"{case_id}.h5"
    write_case_hdf5(case, case_path)
    return [
        {
            "case_id": case_id,
            "path": str(case_path),
            "source_path": str(source),
            "source_dataset": "kWave_channel_mat",
            "shape": list(sound_speed_small.shape),
            "conversion": case.metadata["conversion"],
            "feature_provenance": case.metadata["feature_provenance"],
            "measurement_limitations": case.metadata["measurement_limitations"],
            "has_measured_attenuation": False,
            "has_simulated_attenuation": True,
            "attenuation_evidence": "simulated_ground_truth_line_integral",
        }
    ]


def kwave_channel_mat_metadata(mat_path: str | Path) -> dict[str, Any] | None:
    """Return metadata for a supported k-Wave channel MAT file if present."""

    h5py = _h5py()
    try:
        with h5py.File(mat_path, "r") as handle:
            return _kwave_channel_metadata_from_handle(handle)
    except Exception:
        return None


def _speed_array_to_case(
    sound_speed_mps: np.ndarray,
    *,
    case_id: str,
    source: Path,
    dataset_name: str,
    source_index: int,
    source_shape: tuple[int, int, int],
    spacing_m: tuple[float, float],
    n_transducers: int,
    reference_sound_speed_mps: float,
) -> USCTCase:
    grid = make_grid(shape=tuple(int(v) for v in sound_speed_mps.shape), spacing_m=spacing_m)
    radius_m = 0.6 * max(grid.shape[0] * grid.spacing_m[0], grid.shape[1] * grid.spacing_m[1])
    geometry = make_ring_geometry(n_transducers=n_transducers, radius_m=radius_m)
    projector = StraightRayProjector.from_grid_geometry(grid, geometry)
    delta_slowness = (1.0 / sound_speed_mps) - (1.0 / reference_sound_speed_mps)
    delta_tof_s = projector.forward(delta_slowness).reshape(projector.ray_shape)
    valid_mask = ~np.eye(projector.ray_shape[0], projector.ray_shape[1], dtype=bool)
    log_amp = np.zeros(projector.ray_shape, dtype=float)
    return USCTCase(
        case_id=case_id,
        grid=grid,
        geometry=geometry,
        measurement=MeasurementSpec(
            domain="features",
            delta_tof_s=delta_tof_s,
            log_amp=log_amp,
            valid_mask=valid_mask,
        ),
        ground_truth=GroundTruthSpec(sound_speed_mps=sound_speed_mps),
        metadata={
            "source_dataset": "OpenBreastUS",
            "source_path": str(source),
            "source_mat_dataset": dataset_name,
            "source_index": source_index,
            "source_shape": list(source_shape),
            "conversion": "speed_map_to_straight_ray_surrogate",
            "feature_provenance": "surrogate_delta_tof_from_ground_truth_sound_speed",
            "measurement_domain": "features",
            "measurement_limitations": [
                "source file contains sound-speed maps only",
                "delta_tof_s was generated with a straight-ray projector",
                "log_amp is a zero surrogate and must not be interpreted as measured attenuation",
                "synthetic ring geometry was generated because measured transducer geometry was unavailable",
            ],
            "reference_sound_speed_mps": reference_sound_speed_mps,
            "spacing_m_assumption": list(spacing_m),
            "attenuation_note": "log_amp is a zero surrogate because this source file contains speed maps only.",
        },
    )


def _kwave_arrays_to_case(
    sound_speed_mps: np.ndarray,
    attenuation_np_per_m: np.ndarray,
    *,
    positions_xy: np.ndarray,
    time_s: np.ndarray,
    source: Path,
    source_label: str,
    source_shape: tuple[int, int],
    full_dataset_shape: tuple[int, ...],
    source_npy_path: str | None,
    frequency_hz: float | None,
    xi: np.ndarray | None,
    yi: np.ndarray | None,
    n_transducers: int,
) -> USCTCase:
    grid = _grid_from_coordinates(sound_speed_mps.shape, xi=xi, yi=yi)
    geometry = _geometry_from_xy_positions(positions_xy, n_transducers=n_transducers)
    projector = StraightRayProjector.from_grid_geometry(grid, geometry)
    c0 = float(np.nanmedian(sound_speed_mps))
    delta_slowness = (1.0 / sound_speed_mps) - (1.0 / c0)
    delta_tof_s = projector.forward(delta_slowness).reshape(projector.ray_shape)
    attenuation_integral = projector.forward(attenuation_np_per_m).reshape(projector.ray_shape)
    valid_mask = ~np.eye(projector.ray_shape[0], projector.ray_shape[1], dtype=bool)
    metadata = {
        "source_dataset": "kWave_USCT_simulation",
        "source_path": str(source),
        "source_label": source_label,
        "source_npy_path": source_npy_path,
        "source_shape": list(source_shape),
        "full_dataset_shape": list(full_dataset_shape),
        "conversion": "kwave_channel_mat_to_feature_case",
        "feature_provenance": "surrogate_delta_tof_from_sound_speed_and_attenuation_line_integral_from_simulated_ground_truth",
        "measurement_domain": "features",
        "measurement_limitations": [
            "source file is a k-Wave simulation MAT, not raw OpenBreastUS measured RF data",
            "delta_tof_s was generated with a straight-ray projector from the simulated sound-speed map",
            "log_amp was generated as a straight-ray line integral from the simulated attenuation map",
            "source channel waveforms are present but are not copied into the standard smoke HDF5 case",
        ],
        "reference_sound_speed_mps": c0,
        "attenuation_evidence": "simulated_ground_truth_line_integral",
        "has_simulated_attenuation": True,
        "time_range_s": [float(np.nanmin(time_s)), float(np.nanmax(time_s))] if time_s.size else None,
        "frequency_hz": frequency_hz,
    }
    return USCTCase(
        case_id=_safe_case_id(source_label),
        grid=grid,
        geometry=geometry,
        measurement=MeasurementSpec(
            domain="features",
            frequencies_hz=np.asarray([frequency_hz], dtype=float) if frequency_hz and frequency_hz > 0 else None,
            delta_tof_s=delta_tof_s,
            log_amp=-attenuation_integral,
            valid_mask=valid_mask,
        ),
        ground_truth=GroundTruthSpec(sound_speed_mps=sound_speed_mps, attenuation_np_per_m=attenuation_np_per_m),
        metadata=metadata,
    )


def _downsample_mean(image: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """Downsample by centered crop plus block averaging."""

    ny, nx = image.shape
    out_y, out_x = output_shape
    if out_y <= 0 or out_x <= 0:
        raise ValueError("output_shape must be positive")
    if out_y > ny or out_x > nx:
        y_idx = np.linspace(0, ny - 1, out_y).round().astype(int)
        x_idx = np.linspace(0, nx - 1, out_x).round().astype(int)
        return image[np.ix_(y_idx, x_idx)].astype(float, copy=False)

    block_y = max(1, ny // out_y)
    block_x = max(1, nx // out_x)
    crop_y = out_y * block_y
    crop_x = out_x * block_x
    start_y = (ny - crop_y) // 2
    start_x = (nx - crop_x) // 2
    cropped = image[start_y : start_y + crop_y, start_x : start_x + crop_x]
    return cropped.reshape(out_y, block_y, out_x, block_x).mean(axis=(1, 3))


def _grid_from_coordinates(shape: tuple[int, int], *, xi: np.ndarray | None, yi: np.ndarray | None) -> GridSpec:
    if xi is not None and yi is not None and xi.size > 1 and yi.size > 1:
        x_min, x_max = float(np.nanmin(xi)), float(np.nanmax(xi))
        y_min, y_max = float(np.nanmin(yi)), float(np.nanmax(yi))
        dy = (y_max - y_min) / float(shape[0])
        dx = (x_max - x_min) / float(shape[1])
        if dy > 0.0 and dx > 0.0:
            return GridSpec(shape=shape, spacing_m=(dy, dx), origin_m=(y_min, x_min), roi_mask=np.ones(shape, dtype=bool))
    return make_grid(shape=shape, spacing_m=(1.0e-3, 1.0e-3))


def _geometry_from_xy_positions(positions_xy: np.ndarray, *, n_transducers: int) -> GeometrySpec:
    positions_xy = np.asarray(positions_xy, dtype=float)
    if positions_xy.ndim != 2 or positions_xy.shape[1] != 2:
        raise ValueError("transducerPositionsXY must have shape (n, 2)")
    if n_transducers <= 0 or n_transducers > positions_xy.shape[0]:
        n_transducers = positions_xy.shape[0]
    indices = np.linspace(0, positions_xy.shape[0] - 1, n_transducers).round().astype(int)
    positions_yx = positions_xy[indices][:, [1, 0]]
    radius_m = float(np.median(np.linalg.norm(positions_yx, axis=1)))
    return GeometrySpec(type="ring", tx_pos_m=positions_yx, rx_pos_m=positions_yx.copy(), radius_m=radius_m)


def _kwave_channel_metadata_from_handle(handle: Any) -> dict[str, Any] | None:
    required = ("C", "atten", "full_dataset", "transducerPositionsXY")
    if not all(name in handle for name in required):
        return None
    c_shape = tuple(int(v) for v in handle["C"].shape)
    atten_shape = tuple(int(v) for v in handle["atten"].shape)
    full_shape = tuple(int(v) for v in handle["full_dataset"].shape)
    pos_shape = tuple(int(v) for v in handle["transducerPositionsXY"].shape)
    if len(c_shape) != 2 or atten_shape != c_shape:
        return None
    if len(full_shape) != 3 or len(pos_shape) != 2 or pos_shape[1] != 2:
        return None
    return {
        "format": "kwave-channel-mat",
        "sound_speed_dataset": "C",
        "attenuation_dataset": "atten",
        "channel_dataset": "full_dataset",
        "geometry_dataset": "transducerPositionsXY",
        "sound_speed_shape": list(c_shape),
        "attenuation_shape": list(atten_shape),
        "channel_shape": list(full_shape),
        "geometry_shape": list(pos_shape),
    }


def _decode_matlab_chars(dataset: Any) -> str | None:
    if dataset is None:
        return None
    try:
        values = np.asarray(dataset[()]).reshape(-1)
        return "".join(chr(int(value)) for value in values if int(value) != 0)
    except Exception:
        return None


def _read_scalar(dataset: Any) -> float | None:
    if dataset is None:
        return None
    try:
        return float(np.asarray(dataset[()]).reshape(-1)[0])
    except Exception:
        return None


def _safe_case_id(value: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "kwave_case"


def _largest_3d_dataset_name(handle: Any) -> str:
    candidates: list[tuple[int, str]] = []

    def visitor(name: str, obj: Any) -> None:
        if hasattr(obj, "shape") and len(obj.shape) == 3:
            size = int(np.prod(obj.shape))
            candidates.append((size, name))

    handle.visititems(visitor)
    if not candidates:
        raise ValueError("no 3-D dataset found in MAT/HDF5 file")
    return sorted(candidates, reverse=True)[0][1]


def _h5py():
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError("h5py is required for MAT/HDF5 conversion") from exc
    return h5py
