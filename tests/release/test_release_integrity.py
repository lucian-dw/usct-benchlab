from __future__ import annotations

import subprocess
from pathlib import Path


def test_release_tree_has_expected_slim_shape():
    expected = [
        "src/usctbench/core/schema.py",
        "src/usctbench/core/io.py",
        "src/usctbench/algorithms/ray.py",
        "src/usctbench/algorithms/bent_ray.py",
        "src/usctbench/algorithms/rwave.py",
        "src/usctbench/algorithms/fwi/adapter.py",
        "configs/algorithms/cgls.yaml",
        "configs/benchmarks/synthetic_demo.yaml",
        ".github/workflows/tests.yml",
        "scripts/audit_release.py",
        "docs/math_formulation.md",
        "docs/usage.md",
        "docs/algorithms.md",
        "docs/datasets.md",
        "docs/fwi.md",
        "docs/development.md",
    ]
    forbidden = [
        "src/usctbench/features",
        "src/usctbench/sim",
        "src/usctbench/adapters",
        "configs/benchmarks/archive",
        "configs/algorithms/experimental",
        "scripts/experimental",
        "scripts/matlab_adapters",
    ]

    assert all(Path(path).exists() for path in expected)
    assert not any(Path(path).exists() for path in forbidden)


def test_no_large_scientific_files_are_tracked():
    proc = subprocess.run(
        ["git", "ls-files"], text=True, stdout=subprocess.PIPE, check=True
    )
    forbidden_suffixes = (
        ".h5",
        ".hdf5",
        ".mat",
        ".npy",
        ".npz",
        ".zarr",
        ".pt",
        ".pth",
        ".ckpt",
        ".pkl",
    )
    tracked = [
        path for path in proc.stdout.splitlines() if path.endswith(forbidden_suffixes)
    ]

    assert tracked == []
