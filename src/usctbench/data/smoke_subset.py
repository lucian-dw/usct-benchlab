"""Smoke subset selection for local OpenBreastUS mirrors."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .openbreastus import inspect_openbreastus, write_schema_report


def make_smoke_subset(
    root: str | Path,
    out: str | Path,
    *,
    cases_per_density: int = 1,
    symlink_sources: bool = True,
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
    selected = _select_cases(index["cases"], cases_per_density=cases_per_density)

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

    manifest = {
        "schema_version": "0.1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root_path),
        "subset_root": str(out_path),
        "cases_per_density": cases_per_density,
        "cases": selected,
        "notes": [
            "This smoke subset manifest records selected source files without copying large arrays.",
            "Dataset-specific conversion to standard USCTCase HDF5 should be added after inspecting local schema.",
        ],
    }
    manifest_path = out_path / "openbreastus_smoke_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    write_schema_report(index, out_path / "schema_inspection_report.md")
    return manifest


def _select_cases(cases: list[dict[str, Any]], *, cases_per_density: int) -> list[dict[str, Any]]:
    by_density: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in sorted(cases, key=lambda item: item["case_id"]):
        by_density[case.get("density_class", "unknown")].append(case)

    selected = []
    for density in sorted(by_density):
        selected.extend(by_density[density][:cases_per_density])
    return selected

