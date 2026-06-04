#!/usr/bin/env bash
"exec" "python3" "$0" "$@"
"""Repository-level release audit for the package."""

import subprocess
import sys
from pathlib import Path


REQUIRED = [
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "environment.yml",
    ".gitignore",
    ".github/workflows/tests.yml",
    "src/usctbench/cli.py",
    "src/usctbench/core/schema.py",
    "src/usctbench/core/io.py",
    "src/usctbench/core/registry.py",
    "src/usctbench/algorithms/ray.py",
    "src/usctbench/algorithms/bent_ray.py",
    "src/usctbench/algorithms/rwave.py",
    "src/usctbench/algorithms/fwi/adapter.py",
    "configs/benchmarks/synthetic_demo.yaml",
    "configs/benchmarks/nbpslice2d_demo.yaml",
    "configs/benchmarks/openbreastus_demo.yaml",
    "configs/benchmarks/fwi_kwave_demo.yaml",
    "docs/README.md",
    "docs/math_formulation.md",
    "docs/usage.md",
    "docs/algorithms.md",
    "docs/datasets.md",
    "docs/fwi.md",
    "docs/development.md",
    "docs/references.bib",
    "docs/assets/nbpslice2d_readme_fwi_vs_surrogate.png",
    "docs/assets/openbreastus_readme_fwi_vs_surrogate.png",
]

FORBIDDEN_SUFFIXES = (
    ".h5",
    ".hdf5",
    ".mat",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".ckpt",
    ".pkl",
)
FORBIDDEN_DIRS = [
    "src/usctbench/features",
    "src/usctbench/sim",
    "src/usctbench/adapters",
    "configs/benchmarks/archive",
    "configs/algorithms/experimental",
    "scripts/experimental",
    "scripts/matlab_adapters",
    "docs/internal",
]
FORBIDDEN_FILES = [
    "AGENTS.md",
    "CODEX_GOAL_PROMPT.md",
    "codex_goal_prompt.md",
]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    failures: list[str] = []

    missing = [path for path in REQUIRED if not (root / path).exists()]
    if missing:
        failures.append("missing required files: " + ", ".join(missing))

    existing_forbidden_dirs = [
        path for path in FORBIDDEN_DIRS if (root / path).exists()
    ]
    if existing_forbidden_dirs:
        failures.append("forbidden directories still exist: " + ", ".join(existing_forbidden_dirs))

    existing_forbidden_files = [
        path for path in FORBIDDEN_FILES if (root / path).exists()
    ]
    if existing_forbidden_files:
        failures.append("forbidden development files still exist: " + ", ".join(existing_forbidden_files))

    tracked = _git_ls_files(root)
    data_files = [path for path in tracked if path.endswith(FORBIDDEN_SUFFIXES)]
    if data_files:
        failures.append("tracked data/checkpoint files: " + ", ".join(data_files))

    old_terms = _grep(
        root,
        ["diagnostic-only", "retired", "observable mismatch", "kwave_unified"],
        ["README.md", "configs", "src", "tests"],
    )
    if old_terms:
        failures.append("old internal experiment terms found in user-facing paths")

    personal_patterns = ["/home/" + "wudalong", "/Users/" + "wudalong"]
    personal_paths = _grep(
        root,
        personal_patterns,
        ["README.md", "docs", "configs", "scripts", "src", "tests", ".env.example"],
    )
    if personal_paths:
        failures.append("personal absolute paths found")

    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1
    print("PASS release audit")
    return 0


def _git_ls_files(root: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.strip())
    return proc.stdout.splitlines()


def _grep(root: Path, patterns: list[str], targets: list[str]) -> list[str]:
    results: list[str] = []
    for target in targets:
        path = root / target
        if not path.exists():
            continue
        files = (
            [path]
            if path.is_file()
            else [item for item in path.rglob("*") if item.is_file()]
        )
        for file_path in files:
            try:
                text = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for pattern in patterns:
                if pattern in text:
                    results.append(f"{file_path.relative_to(root)}:{pattern}")
    return results


if __name__ == "__main__":
    raise SystemExit(main())
