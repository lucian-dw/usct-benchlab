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
    "fwi_kwave_adapter": "configs/algorithms/fwi_kwave_adapter.yaml",
}

ALGORITHM_CARDS = {
    "straight_sart": "docs/algorithm_cards/straight_ray_sart.md",
    "straight_sirt": "docs/algorithm_cards/straight_ray_sirt.md",
    "straight_cgls": "docs/algorithm_cards/straight_ray_cgls.md",
    "attenuation_sirt": "docs/algorithm_cards/attenuation_tomography.md",
    "bent_ray_gn": "docs/algorithm_cards/bent_ray_gn.md",
    "rwave_adapter": "docs/algorithm_cards/rwave_ray_born.md",
    "fwi_tiny": "docs/algorithm_cards/fwi_tiny.md",
    "fwi_kwave_adapter": "docs/algorithm_cards/fwi_kwave_adapter.md",
}

REQUIRED_CARD_SECTIONS = [
    "Physical Assumption",
    "Input Requirements",
    "Default Settings",
    "Expected Failure Modes",
    "What To Adjust First",
    "Acceptance Tests",
    "References and Related Code",
]

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
    "scripts/run_v01_release_check.sh",
    "scripts/run_fwi_kwave_adapter_smoke.sh",
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
    "configs/benchmarks/fwi_kwave_adapter_smoke.yaml",
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
    "tests/test_fwi_kwave_adapter.py",
    "tests/test_docs_inventory.py",
    "tests/test_scripts.py",
    "tests/test_v01_audit.py",
]

FORBIDDEN_TRACKED_SUFFIXES = (".h5", ".hdf5", ".mat", ".npy", ".npz", ".pt", ".pth", ".ckpt")
FORBIDDEN_TRACKED_DIRS = ("data/", "runs/", "checkpoints/", "external/", "third_party/")


def audit_repo(
    root: Path,
    *,
    run_dir: Path | None = None,
    openbreastus_index: Path | None = None,
    smoke_manifest: Path | None = None,
    require_clean: bool = False,
    require_v01_dod: bool = False,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    _check_files(root, checks, "required_files", REQUIRED_FILES)
    _check_files(root, checks, "required_tests", REQUIRED_TESTS)
    _check_files(root, checks, "algorithm_configs", list(ALGORITHM_CONFIGS.values()))
    _check_files(root, checks, "algorithm_cards", list(ALGORITHM_CARDS.values()))
    _check_algorithm_card_sections(root, checks)
    _check_registered_algorithms(root, checks)
    _check_optional_adapter_skips(root, checks)
    _check_tracked_data(root, checks)
    if require_clean:
        _check_git_clean(root, checks)
    if run_dir is not None:
        _check_run_dir(run_dir, checks)
    if openbreastus_index is not None:
        _check_openbreastus_index(openbreastus_index, checks)
    if smoke_manifest is not None:
        _check_smoke_manifest(smoke_manifest, checks)
    if require_v01_dod:
        _check_v01_dod_evidence(
            checks,
            run_dir=run_dir,
            openbreastus_index=openbreastus_index,
            smoke_manifest=smoke_manifest,
        )

    passed = all(check["passed"] for check in checks)
    return {"passed": passed, "checks": checks}


def _check_files(root: Path, checks: list[dict[str, Any]], name: str, files: list[str]) -> None:
    missing = [path for path in files if not (root / path).exists()]
    checks.append({"name": name, "passed": not missing, "missing": missing})


def _check_algorithm_card_sections(root: Path, checks: list[dict[str, Any]]) -> None:
    missing_sections: dict[str, list[str]] = {}
    for algorithm, rel_path in ALGORITHM_CARDS.items():
        path = root / rel_path
        if not path.exists():
            missing_sections[algorithm] = REQUIRED_CARD_SECTIONS.copy()
            continue
        headings = {
            line.removeprefix("## ").strip().lower()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("## ")
        }
        missing = [section for section in REQUIRED_CARD_SECTIONS if section.lower() not in headings]
        if missing:
            missing_sections[algorithm] = missing
    checks.append(
        {
            "name": "algorithm_card_sections",
            "passed": not missing_sections,
            "required_sections": REQUIRED_CARD_SECTIONS,
            "missing_sections": missing_sections,
        }
    )


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


def _check_optional_adapter_skips(root: Path, checks: list[dict[str, Any]]) -> None:
    code = r"""
import json
from usctbench.algorithms.adapters.refraction_gn import BentRayGNAdapter
from usctbench.algorithms.adapters.rwave import RWaveAdapter
from usctbench.data.synthetic import make_sound_speed_case
from usctbench.schema import AlgorithmConfig

case = make_sound_speed_case(shape=(8, 8), n_transducers=8)
config = AlgorithmConfig(parameters={"matlab_bin": "/definitely/missing/matlab"})
records = []
for adapter in (BentRayGNAdapter(), RWaveAdapter()):
    result = adapter.run(case, config)
    records.append(
        {
            "algorithm": result.algorithm,
            "status": str(result.status),
            "failure_reason": result.failure_reason or "",
            "adapter_status": result.artifacts.get("adapter_status"),
            "adapter_dependency_available": result.metrics.get("adapter_dependency_available"),
        }
    )
print(json.dumps(records, sort_keys=True))
"""
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
        checks.append({"name": "optional_adapter_skip_evidence", "passed": False, "error": proc.stderr.strip()})
        return
    try:
        records = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        checks.append({"name": "optional_adapter_skip_evidence", "passed": False, "error": f"invalid JSON: {exc}", "stdout": proc.stdout})
        return
    expected = {"bent_ray_gn", "rwave_adapter"}
    observed = {record.get("algorithm") for record in records}
    bad_records = [
        record
        for record in records
        if record.get("status") != "skipped"
        or not record.get("failure_reason")
        or record.get("adapter_status") != "skipped"
        or record.get("adapter_dependency_available") is not False
    ]
    checks.append(
        {
            "name": "optional_adapter_skip_evidence",
            "passed": observed == expected and not bad_records,
            "records": records,
            "missing": sorted(expected - observed),
            "bad_records": bad_records,
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
    run_metadata = run_dir / "run_metadata.yaml"
    missing = [str(path) for path in (summary, report, run_checks_path, run_metadata) if not path.exists()]
    if missing:
        checks.append({"name": "benchmark_run_artifacts", "passed": False, "missing": missing})
        return

    with summary.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    run_checks = json.loads(run_checks_path.read_text(encoding="utf-8"))
    report_text = report.read_text(encoding="utf-8")
    run_fail_reasons = run_checks.get("fail_reasons", [])
    failed_rows = [row for row in rows if row.get("pass") != "True"]
    missing_reason_rows = [row for row in rows if not row.get("pass_reasons") and not row.get("fail_reasons")]
    missing_provenance_fields = [
        field
        for field in ("Git commit:", "Git branch:", "Hostname:", "Python:", "USCT_DATA_ROOT:", "USCT_SAMPLE_ROOT:", "USCT_RUN_ROOT:")
        if field not in report_text
    ]
    checks.append(
        {
            "name": "benchmark_run_artifacts",
            "passed": bool(rows) and not failed_rows and not missing_reason_rows and not run_fail_reasons and not missing_provenance_fields,
            "records": len(rows),
            "failed_rows": failed_rows,
            "missing_reason_rows": missing_reason_rows,
            "run_fail_reasons": run_fail_reasons,
            "missing_provenance_fields": missing_provenance_fields,
        }
    )


def _check_openbreastus_index(index_path: Path, checks: list[dict[str, Any]]) -> None:
    if not index_path.exists():
        checks.append({"name": "openbreastus_index_evidence", "passed": False, "missing": str(index_path)})
        return
    index = json.loads(index_path.read_text(encoding="utf-8"))
    summary = index.get("summary", {})
    cases = index.get("cases", [])
    checks.append(
        {
            "name": "openbreastus_index_evidence",
            "passed": bool(cases) and int(summary.get("num_cases", 0)) > 0,
            "path": str(index_path),
            "num_cases": summary.get("num_cases", 0),
            "num_files": summary.get("num_files", 0),
            "warnings": index.get("warnings", []),
        }
    )


def _check_smoke_manifest(manifest_path: Path, checks: list[dict[str, Any]]) -> None:
    if not manifest_path.exists():
        checks.append({"name": "smoke_manifest_evidence", "passed": False, "missing": str(manifest_path)})
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    converted = manifest.get("converted_cases", [])
    missing_converted = [record.get("path", "") for record in converted if record.get("path") and not Path(record["path"]).exists()]
    checks.append(
        {
            "name": "smoke_manifest_evidence",
            "passed": bool(manifest.get("cases")) and bool(converted) and not missing_converted,
            "path": str(manifest_path),
            "cases": len(manifest.get("cases", [])),
            "converted_cases": len(converted),
            "missing_converted_paths": missing_converted,
        }
    )


def _check_v01_dod_evidence(
    checks: list[dict[str, Any]],
    *,
    run_dir: Path | None,
    openbreastus_index: Path | None,
    smoke_manifest: Path | None,
) -> None:
    missing_inputs = []
    if run_dir is None:
        missing_inputs.append("--run-dir")
    if openbreastus_index is None:
        missing_inputs.append("--openbreastus-index")
    if smoke_manifest is None:
        missing_inputs.append("--smoke-manifest")
    if missing_inputs:
        checks.append({"name": "v01_dod_evidence", "passed": False, "missing_inputs": missing_inputs})
        return

    summary = run_dir / "benchmark_summary.csv"
    if not summary.exists():
        checks.append({"name": "v01_dod_evidence", "passed": False, "missing": str(summary)})
        return
    with summary.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    manifest = json.loads(smoke_manifest.read_text(encoding="utf-8"))
    converted_cases = manifest.get("converted_cases", [])
    converted_case_ids = {str(record.get("case_id")) for record in converted_cases if record.get("case_id")}
    attenuation_capable_case_ids = {
        str(record.get("case_id"))
        for record in converted_cases
        if record.get("case_id") and _has_measured_attenuation_evidence(record)
    }
    required_smoke = {"straight_sart", "attenuation_sirt"}
    passing_smoke = {
        row.get("algorithm")
        for row in rows
        if row.get("algorithm") in required_smoke
        and row.get("status") == "success"
        and row.get("pass") == "True"
        and row.get("case_id") in converted_case_ids
    }
    missing_smoke = sorted(required_smoke - passing_smoke)
    attenuation_passing_cases = {
        str(row.get("case_id"))
        for row in rows
        if row.get("algorithm") == "attenuation_sirt"
        and row.get("status") == "success"
        and row.get("pass") == "True"
        and row.get("case_id") in converted_case_ids
    }
    missing_attenuation_evidence = not bool(attenuation_passing_cases & attenuation_capable_case_ids)
    fail_reasons = []
    if missing_smoke:
        fail_reasons.append(f"missing passing smoke algorithms on converted cases: {', '.join(missing_smoke)}")
    if missing_attenuation_evidence:
        fail_reasons.append("attenuation_sirt has no passing smoke case with measured or simulated nonzero attenuation evidence")
    checks.append(
        {
            "name": "v01_dod_evidence",
            "passed": not fail_reasons,
            "required_smoke_algorithms": sorted(required_smoke),
            "passing_smoke_algorithms": sorted(passing_smoke),
            "missing_smoke_algorithms": missing_smoke,
            "converted_case_ids": sorted(converted_case_ids),
            "attenuation_capable_case_ids": sorted(attenuation_capable_case_ids),
            "attenuation_passing_case_ids": sorted(attenuation_passing_cases),
            "fail_reasons": fail_reasons,
            "run_dir": str(run_dir),
            "openbreastus_index": str(openbreastus_index),
            "smoke_manifest": str(smoke_manifest),
        }
    )


def _has_measured_attenuation_evidence(record: dict[str, Any]) -> bool:
    if record.get("has_measured_attenuation") is True:
        return True
    if record.get("has_simulated_attenuation") is True:
        return True
    evidence = str(record.get("attenuation_evidence", "")).lower()
    if evidence in {
        "measured",
        "measured_log_amp",
        "measured_attenuation",
        "nonzero_log_amp",
        "simulated_ground_truth_line_integral",
        "kwave_channel_peak_log_amp",
    }:
        return True
    limitations = " ".join(str(item) for item in record.get("measurement_limitations", [])).lower()
    if "zero surrogate" in evidence or "zero surrogate" in limitations:
        return False
    if "surrogate" in evidence and "log_amp" in evidence:
        return False
    path = record.get("path")
    if not path:
        return False
    return _hdf5_has_nonzero_log_amp(Path(path))


def _hdf5_has_nonzero_log_amp(path: Path) -> bool:
    try:
        import h5py
    except Exception:
        return False
    try:
        with h5py.File(path, "r") as handle:
            metadata = json.loads(handle.attrs.get("metadata_json", "{}"))
            limitations = " ".join(str(item) for item in metadata.get("measurement_limitations", [])).lower()
            note = str(metadata.get("attenuation_note", "")).lower()
            if "zero surrogate" in limitations or "zero surrogate" in note:
                return False
            if "measurement/log_amp" in handle:
                log_amp = handle["measurement/log_amp"][()]
                return bool((abs(log_amp) > 0).any())
    except Exception:
        return False
    return False


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
    parser.add_argument("--openbreastus-index", default=None, help="Optional OpenBreastUS index JSON evidence path.")
    parser.add_argument("--smoke-manifest", default=None, help="Optional OpenBreastUS smoke manifest JSON evidence path.")
    parser.add_argument("--require-clean", action="store_true", help="Fail if git status is dirty.")
    parser.add_argument("--require-v01-dod", action="store_true", help="Require explicit v0.1 Definition-of-Done evidence paths and smoke records.")
    parser.add_argument("--json", action="store_true", help="Print full JSON output.")
    args = parser.parse_args(argv)

    result = audit_repo(
        Path(args.root).resolve(),
        run_dir=Path(args.run_dir).resolve() if args.run_dir else None,
        openbreastus_index=Path(args.openbreastus_index).resolve() if args.openbreastus_index else None,
        smoke_manifest=Path(args.smoke_manifest).resolve() if args.smoke_manifest else None,
        require_clean=args.require_clean,
        require_v01_dod=args.require_v01_dod,
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
