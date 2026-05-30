"""OpenBreastUS tree inspection helpers.

The real dataset layout can vary across local mirrors. These helpers avoid
hard-coded path assumptions by producing a conservative index from the files
that are actually present.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .conversion import speed_mat_metadata

INSPECTOR_VERSION = "0.1"
SUPPORTED_SUFFIXES = {".h5", ".hdf5", ".mat", ".npy", ".npz", ".json", ".yaml", ".yml", ".txt", ".csv"}
DATA_SUFFIXES = {".h5", ".hdf5", ".mat", ".npy", ".npz"}


def inspect_openbreastus(root: str | Path, out: str | Path | None = None) -> dict[str, Any]:
    """Inspect a local OpenBreastUS tree and optionally write an index JSON."""

    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"OpenBreastUS root does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"OpenBreastUS root is not a directory: {root_path}")

    files = _candidate_files(root_path)
    grouped: dict[str, list[Path]] = defaultdict(list)
    for file_path in files:
        grouped[_case_id(root_path, file_path)].append(file_path)

    cases = [_case_record(root_path, case_id, paths) for case_id, paths in sorted(grouped.items())]
    density_counts: dict[str, int] = defaultdict(int)
    suffix_counts: dict[str, int] = defaultdict(int)
    role_counts: dict[str, int] = defaultdict(int)
    capability_counts: dict[str, int] = defaultdict(int)
    for case in cases:
        density_counts[case["density_class"]] += 1
        for file_record in case["files"]:
            suffix_counts[file_record["suffix"]] += 1
        for role in case["roles"]:
            role_counts[role] += 1
        for name, value in case["capabilities"].items():
            if isinstance(value, bool) and value:
                capability_counts[name] += 1

    warnings = []
    if not cases:
        warnings.append("No candidate OpenBreastUS data files were found under the root.")
    if all(case["density_class"] == "unknown" for case in cases):
        warnings.append("No density classes could be inferred from paths; smoke subset will use unknown density.")

    index = {
        "schema_version": INSPECTOR_VERSION,
        "root": str(root_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "num_cases": len(cases),
            "num_files": len(files),
            "density_counts": dict(sorted(density_counts.items())),
            "suffix_counts": dict(sorted(suffix_counts.items())),
            "role_counts": dict(sorted(role_counts.items())),
            "capability_counts": dict(sorted(capability_counts.items())),
        },
        "cases": cases,
        "warnings": warnings,
    }

    if out is not None:
        out_path = Path(out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    return index


def write_schema_report(index: dict[str, Any], path: str | Path) -> Path:
    """Write a human-readable schema inspection report."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# OpenBreastUS schema inspection",
        "",
        f"- Root: {index.get('root', '')}",
        f"- Cases: {index.get('summary', {}).get('num_cases', 0)}",
        f"- Files: {index.get('summary', {}).get('num_files', 0)}",
        f"- Density counts: {index.get('summary', {}).get('density_counts', {})}",
        "",
        "## Warnings",
        "",
    ]
    warnings = index.get("warnings") or []
    lines.extend(f"- {warning}" for warning in warnings)
    if not warnings:
        lines.append("- None")
    lines.extend(["", "## Case preview", ""])
    for case in index.get("cases", [])[:20]:
        lines.append(
            f"- `{case['case_id']}`: density={case['density_class']}, "
            f"files={len(case['files'])}, roles={case['roles']}, "
            f"capabilities={case.get('capabilities', {})}, frequencies_hz={case['available_frequencies_hz']}"
        )
        for limitation in case.get("limitations", []):
            lines.append(f"  - limitation: {limitation}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _candidate_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES and not _is_hidden(path, root)
    )


def _is_hidden(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    return any(part.startswith(".") for part in rel.parts)


def _case_id(root: Path, file_path: Path) -> str:
    rel = file_path.relative_to(root)
    if len(rel.parts) >= 2:
        return _slug("/".join(rel.parts[:-1]))
    return _slug(file_path.stem)


def _case_record(root: Path, case_id: str, paths: list[Path]) -> dict[str, Any]:
    files = [_file_record(root, path) for path in sorted(paths)]
    roles = sorted({role for file in files for role in file["roles"]})
    frequencies = sorted({freq for file in files for freq in file["frequencies_hz"]})
    capabilities = _case_capabilities(files)
    return {
        "case_id": case_id,
        "density_class": _density_class(paths),
        "split": _split(paths),
        "roles": roles,
        "available_frequencies_hz": frequencies,
        "capabilities": capabilities,
        "limitations": _case_limitations(capabilities),
        "files": files,
    }


def _file_record(root: Path, path: Path) -> dict[str, Any]:
    rel = path.relative_to(root).as_posix()
    schema = _schema(path)
    return {
        "path": rel,
        "suffix": path.suffix.lower(),
        "size_bytes": path.stat().st_size,
        "roles": _roles(path),
        "frequencies_hz": _frequencies_hz(rel),
        "shape": _shape_from_schema(schema),
        "schema": schema,
    }


def _case_capabilities(files: list[dict[str, Any]]) -> dict[str, Any]:
    roles = {role for file in files for role in file["roles"]}
    has_sound_speed = "sound_speed" in roles
    has_attenuation = "attenuation" in roles
    has_wavefield = "wavefield" in roles
    has_reference = "reference" in roles
    has_geometry = "geometry" in roles
    has_mask = "mask" in roles
    has_speed_mat_volume = any(
        "sound_speed" in file["roles"]
        and file["suffix"] == ".mat"
        and isinstance(file.get("schema"), dict)
        and bool(file["schema"].get("largest_3d_dataset"))
        for file in files
    )
    conversion_modes = []
    if has_speed_mat_volume:
        conversion_modes.append("speed_map_to_straight_ray_surrogate")
    if has_wavefield and has_reference and has_geometry:
        conversion_modes.append("frequency_reference_features")
    return {
        "has_sound_speed": has_sound_speed,
        "has_attenuation": has_attenuation,
        "has_wavefield": has_wavefield,
        "has_reference": has_reference,
        "has_geometry": has_geometry,
        "has_mask": has_mask,
        "has_speed_mat_volume": has_speed_mat_volume,
        "convertible_to_usct_case": bool(conversion_modes),
        "conversion_modes": conversion_modes,
    }


def _case_limitations(capabilities: dict[str, Any]) -> list[str]:
    limitations = []
    modes = set(capabilities.get("conversion_modes", []))
    if "speed_map_to_straight_ray_surrogate" in modes and "frequency_reference_features" not in modes:
        limitations.append("speed-map-only case: conversion uses surrogate straight-ray travel-time features, not measured RF/wavefield data")
    if capabilities.get("has_wavefield") and not capabilities.get("has_reference"):
        limitations.append("wavefield-like data found without an identified water/reference file")
    if capabilities.get("has_wavefield") and not capabilities.get("has_geometry"):
        limitations.append("wavefield-like data found without identified geometry/transducer metadata")
    if not capabilities.get("convertible_to_usct_case"):
        limitations.append("no supported automatic USCTCase conversion mode was identified")
    return limitations


def _density_class(paths: list[Path]) -> str:
    text = " ".join(path.as_posix().lower() for path in paths)
    patterns = [
        ("dense", ("dense", "density_high", "high_density", "heterogeneous")),
        ("fatty", ("fatty", "adipose", "density_low", "low_density")),
        ("scattered", ("scattered", "medium_density", "density_medium")),
        ("homogeneous", ("homogeneous", "uniform")),
    ]
    for label, needles in patterns:
        if any(needle in text for needle in needles):
            return label
    match = re.search(r"density[_-]?([a-z0-9]+)", text)
    if match:
        return match.group(1)
    return "unknown"


def _split(paths: list[Path]) -> str:
    text = " ".join(path.as_posix().lower() for path in paths)
    for split in ("train", "training", "val", "valid", "validation", "test", "mini", "smoke"):
        if f"/{split}/" in text or f"_{split}_" in text:
            return "val" if split in {"valid", "validation"} else "train" if split == "training" else split
    return "unknown"


def _roles(path: Path) -> list[str]:
    text = path.as_posix().lower()
    roles = []
    if any(token in text for token in ("reference", "water", "baseline", "background")):
        roles.append("reference")
    if any(token in text for token in ("sound_speed", "soundspeed", "speed", "sos", "c0")):
        roles.append("sound_speed")
    if any(token in text for token in ("attenuation", "atten", "alpha")):
        roles.append("attenuation")
    if any(token in text for token in ("wavefield", "pressure", "rf", "signal", "data")):
        roles.append("wavefield")
    if any(token in text for token in ("geometry", "transducer", "probe", "sensor", "receiver")):
        roles.append("geometry")
    if "mask" in text or "roi" in text:
        roles.append("mask")
    if path.suffix.lower() in DATA_SUFFIXES and not roles:
        roles.append("unknown_data")
    return sorted(set(roles)) or ["metadata"]


def _frequencies_hz(text: str) -> list[float]:
    frequencies = []
    for value, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([kKmM]?[hH][zZ])", text):
        scale = 1.0
        unit_lower = unit.lower()
        if unit_lower.startswith("k"):
            scale = 1.0e3
        elif unit_lower.startswith("m"):
            scale = 1.0e6
        frequencies.append(float(value) * scale)
    return sorted(set(frequencies))


def _schema(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        try:
            array = np.load(path, mmap_mode="r")
            return {"format": "npy", "shape": list(array.shape), "dtype": str(array.dtype)}
        except Exception as exc:
            return {"format": "npy", "read_error": f"{type(exc).__name__}: {exc}"}
    if suffix == ".npz":
        try:
            with np.load(path) as archive:
                return {
                    "format": "npz",
                    "arrays": {name: {"shape": list(archive[name].shape), "dtype": str(archive[name].dtype)} for name in archive.files},
                }
        except Exception as exc:
            return {"format": "npz", "read_error": f"{type(exc).__name__}: {exc}"}
    if suffix in {".h5", ".hdf5", ".mat"}:
        try:
            import h5py
        except ModuleNotFoundError as exc:
            return {"format": "hdf5" if suffix != ".mat" else "mat", "read_error": f"ModuleNotFoundError: {exc}"}
        try:
            datasets: dict[str, dict[str, Any]] = {}
            with h5py.File(path, "r") as handle:
                def visitor(name: str, obj: Any) -> None:
                    if hasattr(obj, "shape") and obj.shape is not None:
                        datasets[name] = {
                            "shape": list(obj.shape),
                            "dtype": str(getattr(obj, "dtype", "")),
                            "ndim": len(obj.shape),
                            "roles": _roles(Path(name)),
                        }

                handle.visititems(visitor)
            schema: dict[str, Any] = {
                "format": "hdf5" if suffix != ".mat" else "mat-v7.3-hdf5",
                "datasets": datasets,
            }
            try:
                schema["largest_3d_dataset"] = speed_mat_metadata(path)
            except Exception:
                schema["largest_3d_dataset"] = None
            return schema
        except Exception as exc:
            return {"format": "hdf5" if suffix != ".mat" else "mat_or_unsupported", "read_error": f"{type(exc).__name__}: {exc}"}
    return {"format": suffix.lstrip(".") or "unknown"}


def _shape_from_schema(schema: dict[str, Any]) -> Any:
    if "shape" in schema:
        return schema["shape"]
    if "arrays" in schema:
        return {name: value["shape"] for name, value in schema["arrays"].items()}
    if "datasets" in schema:
        return {name: value["shape"] for name, value in schema["datasets"].items()}
    if "largest_3d_dataset" in schema:
        return schema["largest_3d_dataset"]
    return None


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    slug = slug.strip("_")
    return slug or "case"
