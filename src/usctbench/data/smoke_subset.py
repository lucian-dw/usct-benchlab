"""Smoke subset selection for local OpenBreastUS mirrors."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .conversion import convert_kwave_channel_mat, convert_speed_mat_volume
from .openbreastus import inspect_openbreastus, write_schema_report


def make_smoke_subset(
    root: str | Path,
    out: str | Path,
    *,
    cases_per_density: int = 1,
    symlink_sources: bool = True,
    convert_speed_mat: bool = True,
    converted_shape: tuple[int, int] = (64, 64),
    spacing_m: tuple[float, float] = (1.0e-3, 1.0e-3),
    n_transducers: int = 32,
    subset_role: str = "interface_smoke",
) -> dict[str, Any]:
    """Select a small smoke subset and write a manifest plus source links.

    The function intentionally avoids copying large source arrays. It creates
    symlinks to selected source files when possible and records all paths in a
    JSON manifest. Dataset-specific conversion to `USCTCase` HDF5 can build on
    this manifest once the actual local schema is known.
    """

    if cases_per_density <= 0:
        raise ValueError("cases_per_density must be positive")

    root_path = Path(root).expanduser().resolve()
    out_path = Path(out).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    index = inspect_openbreastus(root_path, out_path / "openbreastus_index.json")
    selected = _select_cases(index["cases"], cases_per_density=cases_per_density, subset_role=subset_role)

    source_root = out_path / "sources"
    if symlink_sources:
        source_root.mkdir(parents=True, exist_ok=True)
        for case in selected:
            case_dir = source_root / case["case_id"]
            case_dir.mkdir(parents=True, exist_ok=True)
            for file_record in case["files"]:
                source = root_path / file_record["path"]
                link = case_dir / Path(file_record["path"]).name
                if link.exists() or link.is_symlink():
                    link.unlink()
                try:
                    link.symlink_to(os.path.relpath(source, link.parent))
                    file_record["smoke_link"] = str(link.relative_to(out_path))
                except OSError:
                    file_record["smoke_link"] = None

    converted_cases = []
    if convert_speed_mat:
        converted_root = out_path / "cases"
        _clear_converted_cases(converted_root)
        for case in selected:
            for file_record in case["files"]:
                source = root_path / file_record["path"]
                if _is_kwave_channel_mat(file_record):
                    converted_cases.extend(
                        convert_kwave_channel_mat(
                            source,
                            converted_root,
                            case_id_prefix=case["case_id"],
                            output_shape=converted_shape,
                            n_transducers=n_transducers,
                        )
                    )
                elif _is_speed_mat_volume(file_record):
                    speed_indices, speed_index_metadata = _speed_volume_selection(
                        file_record,
                        cases_per_density=cases_per_density,
                        subset_role=subset_role,
                    )
                    converted_cases.extend(
                        convert_speed_mat_volume(
                            source,
                            converted_root,
                            indices=speed_indices,
                            index_metadata=speed_index_metadata,
                            case_id_prefix=case["case_id"],
                            output_shape=converted_shape,
                            spacing_m=spacing_m,
                            n_transducers=n_transducers,
                        )
                    )

    manifest = {
        "schema_version": "0.1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root_path),
        "subset_root": str(out_path),
        "cases_per_density": cases_per_density,
        "converted_shape": list(converted_shape),
        "n_transducers": n_transducers,
        "subset_role": subset_role,
        "cases": selected,
        "case_capability_summary": _capability_summary(selected),
        "converted_cases": converted_cases,
        "notes": [
            "This smoke subset manifest records selected source files without copying large arrays.",
            "Converted HDF5 cases are downsampled standard USCTCase files when a supported speed-map MAT volume is present.",
            "Speed-only conversions use surrogate straight-ray features and record unit assumptions in metadata.",
        ],
    }
    manifest_path = out_path / "openbreastus_smoke_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    write_schema_report(index, out_path / "schema_inspection_report.md")
    return manifest


def make_quality_subset(
    root: str | Path,
    out: str | Path,
    *,
    cases_per_density: int = 1,
    symlink_sources: bool = True,
    convert_speed_mat: bool = True,
    converted_shape: tuple[int, int] = (256, 256),
    spacing_m: tuple[float, float] = (1.0e-3, 1.0e-3),
    n_transducers: int = 128,
) -> dict[str, Any]:
    """Create OpenBreastUS map-surrogate cases for visual quality comparison."""

    return make_smoke_subset(
        root,
        out,
        cases_per_density=cases_per_density,
        symlink_sources=symlink_sources,
        convert_speed_mat=convert_speed_mat,
        converted_shape=converted_shape,
        spacing_m=spacing_m,
        n_transducers=n_transducers,
        subset_role="quality_comparison",
    )


def _capability_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    convertible = [case["case_id"] for case in cases if case.get("capabilities", {}).get("convertible_to_usct_case")]
    limitations = {case["case_id"]: case.get("limitations", []) for case in cases if case.get("limitations")}
    modes: dict[str, int] = defaultdict(int)
    for case in cases:
        for mode in case.get("capabilities", {}).get("conversion_modes", []):
            modes[mode] += 1
    return {
        "convertible_cases": convertible,
        "conversion_mode_counts": dict(sorted(modes.items())),
        "case_limitations": limitations,
    }


def _select_cases(cases: list[dict[str, Any]], *, cases_per_density: int, subset_role: str) -> list[dict[str, Any]]:
    by_density: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in sorted(cases, key=lambda item: item["case_id"]):
        by_density[case.get("density_class", "unknown")].append(case)

    selected = []
    for density in sorted(by_density):
        ranked = sorted(
            by_density[density],
            key=lambda item: (-_conversion_priority(item, subset_role=subset_role), item["case_id"]),
        )
        selected.extend(ranked[:cases_per_density])
    return selected


def _conversion_priority(case: dict[str, Any], *, subset_role: str) -> int:
    modes = set(case.get("capabilities", {}).get("conversion_modes", []))
    if subset_role == "quality_comparison" and _has_canonical_openbreastus_class_volume(case):
        return 40
    if "kwave_channel_mat_to_feature_case" in modes:
        return 30
    if "frequency_reference_features" in modes:
        return 20
    if "speed_map_to_straight_ray_surrogate" in modes:
        return 10
    return 0


def _speed_volume_selection(
    file_record: dict[str, Any],
    *,
    cases_per_density: int,
    subset_role: str,
) -> tuple[list[int], dict[int, dict[str, Any]]]:
    class_ranges = _canonical_openbreastus_class_ranges(file_record)
    if subset_role != "quality_comparison" or class_ranges is None:
        return [0], {}

    indices: list[int] = []
    index_metadata: dict[int, dict[str, Any]] = {}
    for class_id, class_start, class_stop in class_ranges:
        for offset in range(min(cases_per_density, class_stop - class_start)):
            index = class_start + offset
            indices.append(index)
            index_metadata[index] = {
                "case_id_suffix": f"class_{class_id}_{index:06d}",
                "density_class": f"openbreastus_class_{class_id}",
                "density_label": str(class_id),
                "openbreastus_class_id": int(class_id),
                "openbreastus_class_index": int(offset),
                "openbreastus_class_source": "canonical_train_speed_volume_ranges",
            }
    return indices, index_metadata


def _has_canonical_openbreastus_class_volume(case: dict[str, Any]) -> bool:
    return any(
        _canonical_openbreastus_class_ranges(file_record) is not None
        for file_record in case.get("files", [])
    )


def _canonical_openbreastus_class_ranges(file_record: dict[str, Any]) -> list[tuple[int, int, int]] | None:
    """Infer OpenBreastUS four-class ranges for canonical train/test speed volumes.

    The public speed-map mirror is commonly stored as `breast_train_speed.mat`
    with one 3-D dataset, ordered by class in four contiguous blocks. The A100
    preprocessing records confirm this for the current 7200-sample train file.
    This inference is only used for quality panels; measured-data loaders should
    still prefer explicit class metadata when available.
    """

    if not _is_speed_mat_volume(file_record):
        return None
    path = str(file_record.get("path", "")).lower()
    largest = file_record.get("schema", {}).get("largest_3d_dataset") or {}
    dataset = str(largest.get("dataset", "")).lower()
    if "breast_" not in path and not dataset.startswith("breast_"):
        return None
    shape = largest.get("shape") or []
    sample_axis = largest.get("sample_axis")
    if sample_axis is None or len(shape) != 3:
        return None
    case_count = int(shape[int(sample_axis)])
    if case_count < 4 or case_count % 4 != 0:
        return None
    class_size = case_count // 4
    return [(class_id, (class_id - 1) * class_size, class_id * class_size) for class_id in range(1, 5)]


def _is_kwave_channel_mat(file_record: dict[str, Any]) -> bool:
    return bool(file_record.get("schema", {}).get("kwave_channel_mat"))


def _is_speed_mat_volume(file_record: dict[str, Any]) -> bool:
    return (
        "sound_speed" in file_record.get("roles", [])
        and file_record.get("suffix") == ".mat"
        and bool(file_record.get("schema", {}).get("largest_3d_dataset"))
        and not _is_kwave_channel_mat(file_record)
    )


def _clear_converted_cases(converted_root: Path) -> None:
    if not converted_root.exists():
        return
    for path in converted_root.glob("*.h5"):
        if path.is_file() or path.is_symlink():
            path.unlink()
