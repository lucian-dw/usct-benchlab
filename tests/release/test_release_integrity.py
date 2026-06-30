from __future__ import annotations

import importlib.util
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
