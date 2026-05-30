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
    for case in cases:
        density_counts[case["density_class"]] += 1

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
            f"files={len(case['files'])}, roles={case['roles']}, frequencies_hz={case['available_frequencies_hz']}"
        )
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
    return {
        "case_id": case_id,
        "density_class": _density_class(paths),
        "split": _split(paths),
        "roles": roles,
        "available_frequencies_hz": frequencies,
        "files": files,
    }


def _file_record(root: Path, path: Path) -> dict[str, Any]:
    rel = path.relative_to(root).as_posix()
    return {
        "path": rel,
        "suffix": path.suffix.lower(),
        "size_bytes": path.stat().st_size,
        "roles": _roles(path),
        "frequencies_hz": _frequencies_hz(rel),
        "shape": _shape(path),
    }


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


def _shape(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        try:
            array = np.load(path, mmap_mode="r")
            return list(array.shape)
        except Exception:
            return None
    if suffix == ".npz":
        try:
            with np.load(path) as archive:
                return {name: list(archive[name].shape) for name in archive.files}
        except Exception:
            return None
    if suffix in {".h5", ".hdf5"}:
        try:
            import h5py
        except ModuleNotFoundError:
            return None
        try:
            shapes: dict[str, list[int]] = {}
            with h5py.File(path, "r") as handle:
                def visitor(name: str, obj: Any) -> None:
                    if hasattr(obj, "shape") and obj.shape is not None:
                        shapes[name] = list(obj.shape)

                handle.visititems(visitor)
            return shapes
        except Exception:
            return None
    return None


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    slug = slug.strip("_")
    return slug or "case"
