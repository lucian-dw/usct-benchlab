"""Dataset-specific conversion helpers for standard USCTCase files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from usctbench.algorithms.ray.straight_projector import StraightRayProjector
from usctbench.data.synthetic import make_grid, make_ring_geometry
from usctbench.io.hdf5 import write_case_hdf5
from usctbench.schema import GroundTruthSpec, MeasurementSpec, USCTCase


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
