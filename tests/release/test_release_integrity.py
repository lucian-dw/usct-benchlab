from __future__ import annotations

import importlib.util
import json
import re
import subprocess
from pathlib import Path


def _load_audit_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "audit_release.py"
    spec = importlib.util.spec_from_file_location("audit_release", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = _load_audit_module()


def test_release_tree_has_expected_slim_shape():
    assert all(Path(path).exists() for path in AUDIT.REQUIRED)
    assert not any(Path(path).exists() for path in AUDIT.FORBIDDEN_DIRS)
    assert not any(Path(path).exists() for path in AUDIT.FORBIDDEN_FILES)


def test_no_large_scientific_files_are_tracked():
    proc = subprocess.run(
        ["git", "ls-files"], text=True, stdout=subprocess.PIPE, check=True
    )
    tracked = [
        path
        for path in proc.stdout.splitlines()
        if path.endswith(AUDIT.FORBIDDEN_SUFFIXES)
    ]

    assert tracked == []


def test_no_unexpected_large_files_are_tracked():
    proc = subprocess.run(
        ["git", "ls-files"], text=True, stdout=subprocess.PIPE, check=True
    )

    assert AUDIT._large_tracked_files(Path.cwd(), proc.stdout.splitlines()) == []


def test_no_personal_absolute_paths_are_tracked_in_release_targets():
    assert (
        AUDIT._grep_regex(
            Path.cwd(),
            AUDIT.ABSOLUTE_USER_PATH_RE,
            ["README.md", "docs", "configs", "scripts", "src", "tests", ".env.example"],
            allowed_substrings=AUDIT.ABSOLUTE_PATH_ALLOWLIST,
        )
        == []
    )


def test_readme_links_and_reconstruction_assets():
    root = Path(__file__).resolve().parents[2]
    for name in ("README.md", "README.zh-CN.md", "docs/readme_results.md"):
        document = root / name
        for target in re.findall(r"\]\(([^)]+)\)", document.read_text()):
            if "://" in target or target.startswith("#"):
                continue
            assert (document.parent / target.split("#")[0]).exists(), target
    assets = root / "docs/assets/reconstruction"
    manifest = json.loads((assets / "manifest.json").read_text())
    assert manifest["render_only"] is True
    assert manifest["fwi_is_current_wust_validation"] is False
    records = manifest["records"]
    assert len(records) == 96
    assert (
        len({(r["dataset"], r["label"], r["tier"], r["method"]) for r in records}) == 96
    )
    for record in records:
        assert (
            record["image_evaluation"]["used_for_reconstruction_or_stopping"] is False
        )
        assert record["stop_reason"]
        assert (
            record["result_sha256"]
            if record["method"] != "fwi_reference"
            else record["original_result_sha256"]
        )
    for dataset in ("openbreastus", "nbpslice2d"):
        for tier in ("matched", "kwave"):
            path = assets / f"{dataset}_{tier}.png"
            assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
            assert path.stat().st_size < AUDIT.MAX_TRACKED_FILE_BYTES


def test_readme_render_alignment_preserves_axes_and_rejects_extrapolation():
    import numpy as np
    import pytest

    from usctbench.core.schema import GridSpec

    path = Path(__file__).resolve().parents[2] / "scripts/render_readme_results.py"
    spec = importlib.util.spec_from_file_location("readme_renderer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    grid = GridSpec(shape=(3, 4), spacing_m=(0.002, 0.001), origin_m=(0.0, 0.0))
    image = np.arange(12, dtype=float).reshape(3, 4)
    reference = {
        "y_m": (np.arange(3) + 0.5) * 0.002,
        "x_m": (np.arange(4) + 0.5) * 0.001,
    }
    np.testing.assert_allclose(module.align(image, grid, reference), image)
    reference["x_m"] = reference["x_m"] + 0.01
    with pytest.raises(ValueError, match="extends beyond"):
        module.align(image, grid, reference)
