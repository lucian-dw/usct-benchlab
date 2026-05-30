#!/usr/bin/env python
"""Audit v0.1 readiness evidence for usct-benchlab.

This script checks repository artifacts and, optionally, a benchmark run
directory. It is intentionally evidence-oriented: missing files, missing
algorithm cards, missing configs, failed benchmark records, and tracked data
files are reported explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ALGORITHM_CONFIGS = {
    "straight_sart": "configs/algorithms/straight_sart.yaml",
    "straight_sirt": "configs/algorithms/straight_sirt.yaml",
    "straight_cgls": "configs/algorithms/straight_cgls.yaml",
    "attenuation_sirt": "configs/algorithms/attenuation_sirt.yaml",
    "bent_ray_gn": "configs/algorithms/bent_ray_gn.yaml",
    "rwave_adapter": "configs/algorithms/rwave_adapter.yaml",
    "fwi_tiny": "configs/algorithms/fwi_tiny.yaml",
}

ALGORITHM_CARDS = {
    "straight_sart": "docs/algorithm_cards/straight_ray_sart.md",
    "straight_sirt": "docs/algorithm_cards/straight_ray_sirt.md",
    "straight_cgls": "docs/algorithm_cards/straight_ray_cgls.md",
    "attenuation_sirt": "docs/algorithm_cards/attenuation_tomography.md",
    "bent_ray_gn": "docs/algorithm_cards/bent_ray_gn.md",
    "rwave_adapter": "docs/algorithm_cards/rwave_ray_born.md",
    "fwi_tiny": "docs/algorithm_cards/fwi_tiny.md",
}

REQUIRED_FILES = [
    "pyproject.toml",
    "README.md",
    "environment.yml",
    "requirements.txt",
    ".gitignore",
    "src/usctbench/schema.py",
    "src/usctbench/registry.py",
    "src/usctbench/cli.py",
    "src/usctbench/io/hdf5.py",
    "src/usctbench/data/openbreastus.py",
    "src/usctbench/data/smoke_subset.py",
    "src/usctbench/data/features.py",
    "src/usctbench/data/conversion.py",
    "src/usctbench/metrics/image.py",
    "src/usctbench/metrics/data_consistency.py",
    "src/usctbench/benchmark/runner.py",
    "src/usctbench/benchmark/report.py",
    "scripts/setup_workspace.sh",
    "scripts/check_server.sh",
    "scripts/bootstrap_a100.sh",
    "scripts/run_smoke.sh",
    "docs/architecture.md",
    "docs/A100_SERVER_SETUP.md",
    "docs/OPENBREASTUS_DATA_PROTOCOL.md",
    "docs/EVALUATION_ACCEPTANCE_PROTOCOL.md",
    "docs/algorithm_taxonomy.md",
    "docs/EXTERNAL_SOURCES_AND_LICENSES.md",
    "docs/benchmark_report_template.md",
    "docs/ALGORITHM_SETTINGS_TROUBLESHOOTING.md",
    "docs/references.bib",
    "docs/V0_1_READINESS_CHECKLIST.md",
    "configs/benchmarks/openbreastus_smoke.yaml",
    "configs/benchmarks/openbreastus_mini.yaml",
]

REQUIRED_TESTS = [
    "tests/test_schema_roundtrip.py",
    "tests/test_projector_adjoint.py",
    "tests/test_straight_ray_synthetic.py",
    "tests/test_attenuation_synthetic.py",
    "tests/test_metrics.py",
    "tests/test_data_consistency.py",
    "tests/test_features.py",
    "tests/test_openbreastus_inspection.py",
    "tests/test_benchmark_runner.py",
    "tests/test_cli_run_artifacts.py",
    "tests/test_matlab_adapters.py",
    "tests/test_fwi_gradient_check.py",
    "tests/test_fwi_loss_decrease.py",
    "tests/test_docs_inventory.py",
    "tests/test_scripts.py",
    "tests/test_v01_audit.py",
]

FORBIDDEN_TRACKED_SUFFIXES = (".h5", ".hdf5", ".mat", ".npy", ".npz", ".pt", ".pth", ".ckpt")
FORBIDDEN_TRACKED_DIRS = ("data/", "runs/", "checkpoints/", "external/", "third_party/")


def audit_repo(root: Path, *, run_dir: Path | None = None, require_clean: bool = False) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    _check_files(root, checks, "required_files", REQUIRED_FILES)
    _check_files(root, checks, "required_tests", REQUIRED_TESTS)
    _check_files(root, checks, "algorithm_configs", list(ALGORITHM_CONFIGS.values()))
    _check_files(root, checks, "algorithm_cards", list(ALGORITHM_CARDS.values()))
    _check_registered_algorithms(root, checks)
    _check_tracked_data(root, checks)
    if require_clean:
        _check_git_clean(root, checks)
    if run_dir is not None:
        _check_run_dir(run_dir, checks)

    passed = all(check["passed"] for check in checks)
    return {"passed": passed, "checks": checks}


def _check_files(root: Path, checks: list[dict[str, Any]], name: str, files: list[str]) -> None:
    missing = [path for path in files if not (root / path).exists()]
    checks.append({"name": name, "passed": not missing, "missing": missing})


def _check_registered_algorithms(root: Path, checks: list[dict[str, Any]]) -> None:
    code = (
        "from usctbench.cli import register_builtin_algorithms;"
        "from usctbench.registry import list_algorithms;"
        "register_builtin_algorithms();"
        "print('\\n'.join(e.name for e in list_algorithms()))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_pythonpath_env(root),
        check=False,
    )
    if proc.returncode != 0:
        checks.append({"name": "registered_algorithms", "passed": False, "error": proc.stderr.strip()})
        return
    registered = set(proc.stdout.splitlines())
    expected = set(ALGORITHM_CONFIGS)
    missing = sorted(expected - registered)
    checks.append(
        {
            "name": "registered_algorithms",
            "passed": not missing,
            "registered": sorted(registered),
            "missing": missing,
        }
    )


def _check_tracked_data(root: Path, checks: list[dict[str, Any]]) -> None:
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        checks.append({"name": "tracked_data_files", "passed": False, "error": proc.stderr.strip()})
        return
    files = proc.stdout.splitlines()
    forbidden = [
        path
        for path in files
        if path.endswith(FORBIDDEN_TRACKED_SUFFIXES) or any(path == directory.rstrip("/") or path.startswith(directory) for directory in FORBIDDEN_TRACKED_DIRS)
    ]
    checks.append({"name": "tracked_data_files", "passed": not forbidden, "forbidden": forbidden})


def _check_git_clean(root: Path, checks: list[dict[str, Any]]) -> None:
    proc = subprocess.run(
        ["git", "status", "--short"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    dirty = proc.stdout.splitlines()
    checks.append({"name": "git_clean", "passed": proc.returncode == 0 and not dirty, "dirty": dirty})


def _check_run_dir(run_dir: Path, checks: list[dict[str, Any]]) -> None:
    summary = run_dir / "benchmark_summary.csv"
    report = run_dir / "benchmark_report.md"
    run_checks_path = run_dir / "benchmark_run_checks.json"
    missing = [str(path) for path in (summary, report, run_checks_path) if not path.exists()]
    if missing:
        checks.append({"name": "benchmark_run_artifacts", "passed": False, "missing": missing})
        return

    with summary.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    run_checks = json.loads(run_checks_path.read_text(encoding="utf-8"))
    run_fail_reasons = run_checks.get("fail_reasons", [])
    failed_rows = [row for row in rows if row.get("pass") != "True"]
    missing_reason_rows = [row for row in rows if not row.get("pass_reasons") and not row.get("fail_reasons")]
    checks.append(
        {
            "name": "benchmark_run_artifacts",
            "passed": bool(rows) and not failed_rows and not missing_reason_rows and not run_fail_reasons,
            "records": len(rows),
            "failed_rows": failed_rows,
            "missing_reason_rows": missing_reason_rows,
            "run_fail_reasons": run_fail_reasons,
        }
    )


def _pythonpath_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    src = str(root / "src")
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src if not current else f"{src}{os.pathsep}{current}"
    return env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit usct-benchlab v0.1 readiness evidence.")
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument("--run-dir", default=None, help="Optional benchmark run directory to audit.")
    parser.add_argument("--require-clean", action="store_true", help="Fail if git status is dirty.")
    parser.add_argument("--json", action="store_true", help="Print full JSON output.")
    args = parser.parse_args(argv)

    result = audit_repo(
        Path(args.root).resolve(),
        run_dir=Path(args.run_dir).resolve() if args.run_dir else None,
        require_clean=args.require_clean,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for check in result["checks"]:
            status = "PASS" if check["passed"] else "FAIL"
            print(f"{status} {check['name']}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
