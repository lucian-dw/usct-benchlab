"""NBPslices2D numerical breast phantom smoke-subset helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import io
import h5py
import numpy as np
from .conversion import resample_bilinear, resample_labels_nearest, _safe_case_id
from usctbench.data.synthetic import (
    build_surrogate_tof_case,
    make_grid,
    make_ring_geometry,
)
from usctbench.core.io import write_case_hdf5
from usctbench.core.provenance import MeasurementProvenance, stamp_measurement_metadata
from usctbench.core.schema import USCTCase


def inspect_nbp_slice2d_zip(
    zip_path: str | Path, out: str | Path | None = None
) -> dict[str, Any]:
    """Inspect a local NBPslices2D ZIP archive without extracting data."""

    source = Path(zip_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"NBPslices2D ZIP does not exist: {source}")
    with zipfile.ZipFile(source) as archive:
        members = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".mat") and "__MACOSX" not in Path(name).parts
        ]
    label_counts = Counter(Path(name).stem[:1].upper() or "unknown" for name in members)
    index = {
        "schema_version": "0.1",
        "source_zip": str(source),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "num_cases": len(members),
            "density_label_counts": dict(sorted(label_counts.items())),
            "density_class_counts": {
                NBP_DENSITY_CLASSES.get(label, "unknown"): count
                for label, count in sorted(label_counts.items())
            },
        },
        "cases": [
            {
                "member": name,
                "case_id": Path(name).stem,
                "density_label": Path(name).stem[:1].upper(),
                "density_class": NBP_DENSITY_CLASSES.get(
                    Path(name).stem[:1].upper(), "unknown"
                ),
            }
            for name in sorted(members)
        ],
    }
    if out is not None:
        out_path = Path(out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(index, indent=2, sort_keys=True), encoding="utf-8"
        )
    return index


def prepare_nbpslice2d(
    zip_path: str | Path,
    out: str | Path,
    *,
    cases_per_type: int = 1,
    converted_shape: tuple[int, int] = (64, 64),
    n_transducers: int = 32,
    reference_sound_speed_mps: float = 1500.0,
    case_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Prepare selected property maps and model-matched travel-time observations."""

    out_path = Path(out).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    cases_root = out_path / "cases"
    converted_cases = convert_nbp_slice2d_zip(
        zip_path,
        cases_root,
        cases_per_type=cases_per_type,
        case_ids=case_ids,
        output_shape=converted_shape,
        n_transducers=n_transducers,
        reference_sound_speed_mps=reference_sound_speed_mps,
    )
    manifest = {
        "cases": [
            {"case_id": c["case_id"], "path": c["path"]} for c in converted_cases
        ],
        "source_zip": str(Path(zip_path).expanduser().resolve()),
        "settings": {
            "shape": list(converted_shape),
            "n_transducers": n_transducers,
            "reference_sound_speed_mps": reference_sound_speed_mps,
        },
    }
    manifest_path = out_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


NBP_PIXEL_SPACING_M = 1.0e-4
NBP_ROI_FOV_FRACTION = 0.72
NBP_DENSITY_CLASSES = {
    "A": "almost_entirely_fatty",
    "B": "scattered_fibroglandular",
    "C": "heterogeneously_dense",
    "D": "extremely_dense",
}


def convert_nbp_slice2d_mat(
    mat_path: str | Path,
    out_dir: str | Path,
    *,
    case_id_prefix: str | None = None,
    output_shape: tuple[int, int] = (64, 64),
    n_transducers: int = 32,
    reference_sound_speed_mps: float = 1500.0,
) -> list[dict[str, Any]]:
    """Convert one NBPslices2D MAT file to a standard feature-domain case."""

    source = Path(mat_path).expanduser().resolve()
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    with h5py.File(source, "r") as handle:
        case, record = _nbp_handle_to_case_record(
            handle,
            case_id=case_id_prefix or _safe_case_id(source.stem),
            source_path=str(source),
            source_member=None,
            output_shape=output_shape,
            n_transducers=n_transducers,
            reference_sound_speed_mps=reference_sound_speed_mps,
        )
    case_path = out_path / f"{case.case_id}.h5"
    write_case_hdf5(case, case_path)
    record["path"] = str(case_path)
    return [record]


def convert_nbp_slice2d_zip(
    zip_path: str | Path,
    out_dir: str | Path,
    *,
    cases_per_type: int = 1,
    case_ids: list[str] | None = None,
    output_shape: tuple[int, int] = (64, 64),
    n_transducers: int = 32,
    reference_sound_speed_mps: float = 1500.0,
) -> list[dict[str, Any]]:
    """Convert selected NBPslices2D MAT members from a ZIP archive."""

    if cases_per_type <= 0:
        raise ValueError("cases_per_type must be positive")

    source = Path(zip_path).expanduser().resolve()
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    with zipfile.ZipFile(source) as archive:
        selected = _select_nbp_zip_members(
            archive.namelist(), cases_per_type=cases_per_type
        )
        if case_ids is not None:
            wanted = set(case_ids)
            selected = [
                name
                for name in archive.namelist()
                if Path(name).stem in wanted and name.endswith(".mat")
            ]
            missing = wanted - {Path(name).stem for name in selected}
            if missing:
                raise ValueError(f"NBPslice2D cases not found: {sorted(missing)}")
        for member in selected:
            data = archive.read(member)
            with h5py.File(io.BytesIO(data), "r") as handle:
                case_id = _safe_case_id(Path(member).stem)
                case, record = _nbp_handle_to_case_record(
                    handle,
                    case_id=case_id,
                    source_path=str(source),
                    source_member=member,
                    output_shape=output_shape,
                    n_transducers=n_transducers,
                    reference_sound_speed_mps=reference_sound_speed_mps,
                )
            case_path = out_path / f"{case.case_id}.h5"
            write_case_hdf5(case, case_path)
            record["path"] = str(case_path)
            records.append(record)
    return records


def _nbp_handle_to_case_record(
    handle: Any,
    *,
    case_id: str,
    source_path: str,
    source_member: str | None,
    output_shape: tuple[int, int],
    n_transducers: int,
    reference_sound_speed_mps: float,
) -> tuple[USCTCase, dict[str, Any]]:
    required = ("sos", "den", "label", "type")
    if not all(name in handle for name in required):
        raise ValueError("not a supported NBPslices2D MAT file")
    raw = {name: handle[name][()] for name in required}
    sos_mps_raw = np.asarray(raw["sos"], dtype=float) * 1000.0
    label_raw = np.asarray(raw["label"], dtype=np.uint8)
    density_raw = np.asarray(raw["den"], dtype=float)
    if (
        sos_mps_raw.ndim != 2
        or density_raw.shape != sos_mps_raw.shape
        or label_raw.shape != sos_mps_raw.shape
    ):
        raise ValueError("NBPslices2D sos, den and label must have matching 2-D shapes")
    raw_type = np.asarray(raw["type"]).reshape(-1)
    if (
        raw_type.size != 1
        or not np.isfinite(raw_type[0])
        or raw_type[0] != int(raw_type[0])
    ):
        raise ValueError("NBPslices2D type must contain one finite integer code")
    type_code = int(raw_type[0])
    density_label = chr(type_code) if 0 <= type_code <= 255 else str(type_code)
    density_class = NBP_DENSITY_CLASSES.get(density_label, "unknown")

    sos_mps_crop, label_crop, crop_info = _fit_nbp_field_of_view(sos_mps_raw, label_raw)

    sound_speed_mps = resample_bilinear(sos_mps_crop, output_shape)
    label_small = resample_labels_nearest(label_crop, output_shape)
    roi_mask = label_small > 0
    if not np.any(roi_mask):
        roi_mask = None

    spacing_m = (
        NBP_PIXEL_SPACING_M * sos_mps_crop.shape[0] / float(output_shape[0]),
        NBP_PIXEL_SPACING_M * sos_mps_crop.shape[1] / float(output_shape[1]),
    )
    grid = make_grid(shape=output_shape, spacing_m=spacing_m)
    grid = grid.model_copy(update={"roi_mask": roi_mask})
    radius_m = 0.6 * max(
        grid.shape[0] * grid.spacing_m[0], grid.shape[1] * grid.spacing_m[1]
    )
    geometry = make_ring_geometry(n_transducers=n_transducers, radius_m=radius_m)

    source_ref = f"{source_path}!{source_member}" if source_member else source_path
    case = build_surrogate_tof_case(
        grid,
        geometry,
        sound_speed_mps,
        reference_sound_speed_mps,
        case_id,
        source=stamp_measurement_metadata(
            {
                "source_dataset": "NBPslices2D",
                "case_type": "nbpslice2d_property_map_surrogate",
                "source_path": source_path,
                "source_member": source_member,
                "source_ref": source_ref,
                "source_shape": list(sos_mps_raw.shape),
                "fitted_source_shape": list(sos_mps_crop.shape),
                "roi_fit": crop_info,
                "conversion": "nbpslice2d_to_feature_case",
                "feature_provenance": "surrogate_delta_tof_from_nbp_ground_truth_sound_speed",
                "measurement_domain": "features",
                "measurement_limitations": [
                    "NBPslices2D contains acoustic property maps, not measured RF or pressure wavefields",
                    "delta_tof_s was generated with a straight-ray projector from the sound-speed map",
                    "synthetic ring geometry was generated because acquisition geometry is not included in the slice files",
                ],
                "reference_sound_speed_mps": reference_sound_speed_mps,
                "density_class": density_class,
                "density_label": density_label,
                "nbp_type_code": type_code,
                "category_source": "mat_type_field",
                "roi_status": "available" if roi_mask is not None else "missing_label",
                "roi_source": "ground_truth_label" if roi_mask is not None else "none",
                "roi_use": (
                    "reconstruction_prior" if roi_mask is not None else "unrestricted"
                ),
                "source_sound_speed_units": "km/s",
                "source_image_axes": "yx",
                "geometry_source": "synthetic_ring",
                "density_source_units": "g/mm^3",
                "density_kg_per_m3_min": float(np.nanmin(density_raw) * 1.0e9),
                "density_kg_per_m3_max": float(np.nanmax(density_raw) * 1.0e9),
                "label_values": [
                    int(value) for value in sorted(np.unique(label_raw).tolist())
                ],
                "pixel_spacing_m_assumption": NBP_PIXEL_SPACING_M,
                "effective_spacing_m": list(spacing_m),
            },
            measurement_provenance=MeasurementProvenance.SPEEDMAP_TRAVEL_TIME_SURROGATE,
            benchmark_type="speedmap_travel_time_surrogate",
            forward_model="straight_ray_speedmap_surrogate",
            feature_source="surrogate_delta_tof_from_nbp_property_maps",
        ),
    )
    record = {
        "case_id": case_id,
        "source_path": source_path,
        "source_member": source_member,
        "source_dataset": "NBPslices2D",
        "case_type": case.metadata["case_type"],
        "benchmark_type": case.metadata["benchmark_type"],
        "shape": list(sound_speed_mps.shape),
        "source_shape": list(sos_mps_raw.shape),
        "fitted_source_shape": list(sos_mps_crop.shape),
        "roi_fit": crop_info,
        "conversion": case.metadata["conversion"],
        "feature_provenance": case.metadata["feature_provenance"],
        "measurement_limitations": case.metadata["measurement_limitations"],
        "density_label": density_label,
        "density_class": density_class,
    }
    return case, record


def _fit_nbp_field_of_view(
    sound_speed_mps: np.ndarray,
    label: np.ndarray,
    *,
    roi_fov_fraction: float = NBP_ROI_FOV_FRACTION,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Crop an NBPslice2D map so the breast ROI occupies a useful FOV."""

    label = np.asarray(label)
    roi = label > 0
    if not np.any(roi):
        info = {
            "enabled": False,
            "reason": "no_positive_label",
            "source_bbox_pixels": [0, 0, int(label.shape[0]), int(label.shape[1])],
            "crop_bbox_pixels": [0, 0, int(label.shape[0]), int(label.shape[1])],
            "target_roi_fov_fraction": float(roi_fov_fraction),
        }
        return sound_speed_mps, label, info

    rows, cols = np.where(roi)
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(cols.min()), int(cols.max()) + 1
    roi_h = y1 - y0
    roi_w = x1 - x0
    if not 0.0 < roi_fov_fraction <= 1.0:
        raise ValueError("roi_fov_fraction must be in (0, 1]")
    side = int(np.ceil(max(roi_h, roi_w) / roi_fov_fraction))
    side = max(side, roi_h, roi_w, 1)
    side = min(side, int(label.shape[0]), int(label.shape[1]))
    center_y = 0.5 * (y0 + y1)
    center_x = 0.5 * (x0 + x1)
    crop_y0 = int(round(center_y - 0.5 * side))
    crop_x0 = int(round(center_x - 0.5 * side))
    crop_y0 = min(max(crop_y0, 0), int(label.shape[0]) - side)
    crop_x0 = min(max(crop_x0, 0), int(label.shape[1]) - side)
    crop_y1 = crop_y0 + side
    crop_x1 = crop_x0 + side
    info = {
        "enabled": True,
        "source_bbox_pixels": [y0, x0, y1, x1],
        "crop_bbox_pixels": [crop_y0, crop_x0, crop_y1, crop_x1],
        "source_roi_shape_pixels": [roi_h, roi_w],
        "crop_shape_pixels": [side, side],
        "target_roi_fov_fraction": float(roi_fov_fraction),
        "actual_roi_fov_fraction": float(max(roi_h, roi_w) / side),
    }
    crop = np.s_[crop_y0:crop_y1, crop_x0:crop_x1]
    return sound_speed_mps[crop], label[crop], info


def _select_nbp_zip_members(names: list[str], *, cases_per_type: int) -> list[str]:
    grouped: dict[str, list[str]] = {}
    for name in names:
        path = Path(name)
        if path.suffix.lower() != ".mat":
            continue
        if any(part.startswith("__MACOSX") for part in path.parts):
            continue
        label = path.stem[:1].upper() or "unknown"
        grouped.setdefault(label, []).append(name)
    selected: list[str] = []
    for label in sorted(grouped):
        selected.extend(sorted(grouped[label])[:cases_per_type])
    return selected
