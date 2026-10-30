"""Explicit OpenBreastUS file selection and separate format inspection."""

import json
from pathlib import Path

import h5py
from scipy.io import whosmat

from contextlib import contextmanager
from typing import Any
import numpy as np
from .conversion import _downsample_mean
from usctbench.data.synthetic import (
    build_surrogate_tof_case,
    make_grid,
    make_ring_geometry,
)
from usctbench.core.io import write_case_hdf5
from usctbench.core.provenance import MeasurementProvenance, stamp_measurement_metadata
from usctbench.core.schema import USCTCase


def inspect_openbreastus(root):
    """Report stored array names/shapes without inferring physics or classes."""
    root = Path(root).expanduser().resolve(strict=True)
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".mat", ".h5", ".hdf5"}:
            continue
        record = {"path": path.relative_to(root).as_posix(), "arrays": []}
        try:
            if h5py.is_hdf5(path):
                with h5py.File(path, "r") as handle:

                    def visit(name, obj):
                        if isinstance(obj, h5py.Dataset):
                            record["arrays"].append(
                                {
                                    "name": name,
                                    "shape": list(obj.shape),
                                    "dtype": str(obj.dtype),
                                }
                            )

                    handle.visititems(visit)
            else:
                record["arrays"] = [
                    {"name": name, "shape": list(shape), "dtype": dtype}
                    for name, shape, dtype in whosmat(path)
                ]
        except (OSError, ValueError, NotImplementedError) as exc:
            record["read_error"] = f"{type(exc).__name__}: {exc}"
        files.append(record)
    return {"root": str(root), "files": files}


def prepare_openbreastus(
    root,
    out,
    *,
    files,
    sample_indices,
    spacing_m,
    dataset_name,
    sample_axis=2,
    image_axes="yx",
    sound_speed_units="m/s",
    converted_shape=(64, 64),
    n_transducers=32,
    reference_sound_speed_mps=1500.0,
):
    """Convert declared speed volumes/slices to model-generated ToF cases.

    Pressure import is a separate operation. No density labels, sample groups,
    source links or physical units are inferred from file names.
    """
    root = Path(root).expanduser().resolve(strict=True)
    out = Path(out).expanduser().resolve()
    selected = [(root / name).resolve(strict=True) for name in files]
    if len(set(selected)) != len(selected):
        raise ValueError("Duplicate OpenBreastUS input files")
    # Distinct relative names retain their path components in generated IDs.
    prefixes = [
        p.relative_to(root).with_suffix("").as_posix().replace("/", "__")
        for p in selected
    ]
    if len(set(prefixes)) != len(prefixes):
        raise ValueError("Input files would produce duplicate case IDs")
    out.mkdir(parents=True, exist_ok=True)
    cases = []
    for source, prefix in zip(selected, prefixes):
        cases.extend(
            convert_speed_mat_volume(
                source,
                out / "cases",
                dataset_name=dataset_name,
                indices=sample_indices,
                case_id_prefix=prefix,
                output_shape=converted_shape,
                spacing_m=spacing_m,
                n_transducers=n_transducers,
                reference_sound_speed_mps=reference_sound_speed_mps,
                sample_axis=sample_axis,
                image_axes=image_axes,
                sound_speed_units=sound_speed_units,
            )
        )
    manifest = {
        "source_root": str(root),
        "cases": cases,
        "settings": {
            "files": list(files),
            "dataset_name": dataset_name,
            "sample_indices": list(sample_indices),
            "shape": list(converted_shape),
            "spacing_m": list(spacing_m),
            "sample_axis": sample_axis,
            "image_axes": image_axes,
            "sound_speed_units": sound_speed_units,
            "n_transducers": n_transducers,
            "reference_sound_speed_mps": reference_sound_speed_mps,
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def convert_speed_mat_volume(
    mat_path: str | Path,
    out_dir: str | Path,
    *,
    indices: list[int] | None = None,
    index_metadata: dict[int, dict[str, Any]] | None = None,
    dataset_name: str,
    sample_axis: int = 2,
    image_axes: str = "yx",
    sound_speed_units: str = "m/s",
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
    generates straight-ray surrogate travel-time features from the speed map.
    Metadata records these assumptions explicitly.
    """

    source = Path(mat_path).expanduser().resolve()
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    with _open_speed_volume(source, dataset_name=dataset_name) as speed_volume:
        name = speed_volume["name"]
        dataset = speed_volume["dataset"]
        shape = tuple(int(value) for value in dataset.shape)
        if len(shape) != 3:
            raise ValueError(
                f"speed dataset must be 3-D [ny, nx, n_cases] or [n_cases,ny,nx], got {shape}"
            )
        if sample_axis not in (0, 1, 2) or image_axes not in ("yx", "xy"):
            raise ValueError("sample_axis must be 0/1/2 and image_axes must be yx/xy")
        if sound_speed_units not in ("m/s", "km/s"):
            raise ValueError("sound_speed_units must be m/s or km/s")
        case_count = shape[sample_axis]
        selected_indices = indices if indices is not None else [0]
        for index in selected_indices:
            if index < 0 or index >= case_count:
                raise IndexError(f"case index {index} outside dataset shape {shape}")
            extra_metadata = dict((index_metadata or {}).get(index, {}))
            sound_speed = np.asarray(
                _read_speed_volume_slice(dataset, index, sample_axis), dtype=float
            )
            if image_axes == "xy":
                sound_speed = sound_speed.T
            if sound_speed_units == "km/s":
                sound_speed = sound_speed * 1000.0
            sound_speed = _downsample_mean(sound_speed, output_shape)
            case_id_suffix = extra_metadata.pop("case_id_suffix", None)
            prefix = case_id_prefix or source.stem
            case_id = (
                f"{prefix}_{case_id_suffix}"
                if case_id_suffix
                else f"{prefix}_{index:06d}"
            )
            case = _speed_array_to_case(
                sound_speed,
                case_id=case_id,
                source=source,
                dataset_name=name,
                source_index=index,
                source_shape=shape,
                spacing_m=spacing_m,
                n_transducers=n_transducers,
                reference_sound_speed_mps=reference_sound_speed_mps,
            )
            case.metadata.update(
                source_sample_axis=sample_axis,
                source_image_axes=image_axes,
                source_sound_speed_units=sound_speed_units,
                spatial_metadata_source="conversion_parameters",
                geometry_source="synthetic_ring",
                category_source=(
                    extra_metadata.get(
                        "openbreastus_class_source", "provided_index_metadata"
                    )
                    if extra_metadata
                    else "unknown"
                ),
            )
            if extra_metadata:
                case = case.model_copy(
                    update={"metadata": {**case.metadata, **extra_metadata}}
                )
            case_path = out_path / f"{case_id}.h5"
            write_case_hdf5(case, case_path)
            record = {
                "case_id": case_id,
                "path": str(case_path),
                "source_path": str(source),
                "source_dataset": name,
                "source_index": index,
                "shape": list(sound_speed.shape),
                "conversion": case.metadata["conversion"],
                "case_type": case.metadata["case_type"],
                "benchmark_type": case.metadata["benchmark_type"],
                "feature_provenance": case.metadata["feature_provenance"],
                "measurement_limitations": case.metadata["measurement_limitations"],
            }
            if extra_metadata:
                record.update(extra_metadata)
            records.append(record)
    return records


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
    grid = make_grid(
        shape=tuple(int(v) for v in sound_speed_mps.shape), spacing_m=spacing_m
    )
    radius_m = 0.6 * max(
        grid.shape[0] * grid.spacing_m[0], grid.shape[1] * grid.spacing_m[1]
    )
    geometry = make_ring_geometry(n_transducers=n_transducers, radius_m=radius_m)
    return build_surrogate_tof_case(
        grid,
        geometry,
        sound_speed_mps,
        reference_sound_speed_mps,
        case_id,
        source=stamp_measurement_metadata(
            {
                "source_dataset": "OpenBreastUS",
                "case_type": "openbreastus_speedmap_surrogate",
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
                    "synthetic ring geometry was generated because measured transducer geometry was unavailable",
                ],
                "reference_sound_speed_mps": reference_sound_speed_mps,
                "spacing_m_assumption": list(spacing_m),
            },
            measurement_provenance=MeasurementProvenance.SPEEDMAP_TRAVEL_TIME_SURROGATE,
            benchmark_type="speedmap_travel_time_surrogate",
            forward_model="straight_ray_speedmap_surrogate",
            feature_source="surrogate_delta_tof_from_ground_truth_sound_speed",
        ),
    )


@contextmanager
def _open_speed_volume(path: Path, *, dataset_name: str):
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError("dataset_name must explicitly select a sound-speed array")
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as handle:
            name = dataset_name
            yield {"name": name, "dataset": handle[name], "format": "matlab-v7.3-hdf5"}
            return

    try:
        from scipy.io import loadmat
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"{path} is not HDF5; scipy is required to read MATLAB v5 MAT speed volumes"
        ) from exc

    data = loadmat(path)
    name = dataset_name
    if name not in data:
        keys = [key for key in data if not key.startswith("__")]
        raise KeyError(f"{name!r} not found in {path}. Available keys: {keys}")
    yield {"name": name, "dataset": np.asarray(data[name]), "format": "matlab-v5"}


def _read_speed_volume_slice(dataset: Any, index: int, sample_axis: int) -> np.ndarray:
    if sample_axis == 0:
        return np.asarray(dataset[index, :, :])
    if sample_axis == 1:
        return np.asarray(dataset[:, index, :])
    if sample_axis == 2:
        return np.asarray(dataset[:, :, index])
    raise ValueError(f"invalid sample axis {sample_axis}")
