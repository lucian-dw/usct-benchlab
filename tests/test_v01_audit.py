from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_audit_module():
    path = Path("scripts/audit_v01_readiness.py")
    spec = importlib.util.spec_from_file_location("audit_v01_readiness", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_v01_audit_repo_passes_without_run_dir():
    audit = _load_audit_module()

    result = audit.audit_repo(Path(".").resolve())

    assert result["passed"] is True
    assert {check["name"] for check in result["checks"]} >= {
        "required_files",
        "required_tests",
        "algorithm_configs",
        "algorithm_cards",
        "registered_algorithms",
        "tracked_data_files",
    }


def test_v01_audit_run_dir_requires_passing_records(tmp_path):
    audit = _load_audit_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with (run_dir / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["algorithm", "case_id", "pass", "pass_reasons", "fail_reasons"])
        writer.writeheader()
        writer.writerow({"algorithm": "a", "case_id": "c", "pass": "False", "pass_reasons": "", "fail_reasons": "status is failed"})
    (run_dir / "benchmark_report.md").write_text("# Benchmark report\n", encoding="utf-8")
    (run_dir / "benchmark_run_checks.json").write_text(
        json.dumps({"passed": False, "pass_reasons": [], "fail_reasons": ["record count 1 below required 2"]}),
        encoding="utf-8",
    )

    result = audit.audit_repo(Path(".").resolve(), run_dir=run_dir)

    assert result["passed"] is False
    run_check = [check for check in result["checks"] if check["name"] == "benchmark_run_artifacts"][0]
    assert run_check["failed_rows"]
    assert run_check["run_fail_reasons"]
