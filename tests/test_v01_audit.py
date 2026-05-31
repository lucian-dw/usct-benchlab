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


def _write_run_metadata(run_dir: Path) -> None:
    (run_dir / "run_metadata.yaml").write_text(
        "git:\n  commit: abc123\n  branch: test\nhost:\n  hostname: test-host\npython:\n  version: test-python\n",
        encoding="utf-8",
    )


def _write_report(run_dir: Path) -> None:
    (run_dir / "benchmark_report.md").write_text(
        "\n".join(
            [
                "# Benchmark report",
                "- Git commit: abc123",
                "- Git branch: test",
                "- Hostname: test-host",
                "- Python: test-python",
                "- USCT_DATA_ROOT: /data",
                "- USCT_SAMPLE_ROOT: /sample",
                "- USCT_RUN_ROOT: /runs",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_v01_audit_repo_passes_without_run_dir():
    audit = _load_audit_module()

    result = audit.audit_repo(Path(".").resolve())

    assert result["passed"] is True
    assert {check["name"] for check in result["checks"]} >= {
        "required_files",
        "required_tests",
        "algorithm_configs",
        "algorithm_cards",
        "algorithm_card_sections",
        "registered_algorithms",
        "optional_adapter_skip_evidence",
        "tracked_data_files",
    }

    card_check = [check for check in result["checks"] if check["name"] == "algorithm_card_sections"][0]
    assert card_check["passed"] is True
    assert not card_check["missing_sections"]

    adapter_check = [check for check in result["checks"] if check["name"] == "optional_adapter_skip_evidence"][0]
    assert adapter_check["passed"] is True
    assert {record["algorithm"] for record in adapter_check["records"]} == {"bent_ray_gn", "rwave_adapter"}


def test_v01_audit_run_dir_requires_passing_records(tmp_path):
    audit = _load_audit_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with (run_dir / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["algorithm", "case_id", "pass", "pass_reasons", "fail_reasons"])
        writer.writeheader()
        writer.writerow({"algorithm": "a", "case_id": "c", "pass": "False", "pass_reasons": "", "fail_reasons": "status is failed"})
    _write_report(run_dir)
    _write_run_metadata(run_dir)
    (run_dir / "benchmark_run_checks.json").write_text(
        json.dumps({"passed": False, "pass_reasons": [], "fail_reasons": ["record count 1 below required 2"]}),
        encoding="utf-8",
    )

    result = audit.audit_repo(Path(".").resolve(), run_dir=run_dir)

    assert result["passed"] is False
    run_check = [check for check in result["checks"] if check["name"] == "benchmark_run_artifacts"][0]
    assert run_check["failed_rows"]
    assert run_check["run_fail_reasons"]


def test_v01_audit_requires_explicit_dod_evidence_paths():
    audit = _load_audit_module()

    result = audit.audit_repo(Path(".").resolve(), require_v01_dod=True)

    assert result["passed"] is False
    dod_check = [check for check in result["checks"] if check["name"] == "v01_dod_evidence"][0]
    assert set(dod_check["missing_inputs"]) == {"--run-dir", "--openbreastus-index", "--smoke-manifest"}


def test_v01_audit_accepts_complete_dod_evidence(tmp_path):
    audit = _load_audit_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with (run_dir / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["algorithm", "case_id", "status", "pass", "pass_reasons", "fail_reasons"])
        writer.writeheader()
        writer.writerow({"algorithm": "straight_sart", "case_id": "c", "status": "success", "pass": "True", "pass_reasons": "ok", "fail_reasons": ""})
        writer.writerow({"algorithm": "attenuation_sirt", "case_id": "c", "status": "success", "pass": "True", "pass_reasons": "ok", "fail_reasons": ""})
    _write_report(run_dir)
    _write_run_metadata(run_dir)
    (run_dir / "benchmark_run_checks.json").write_text(
        json.dumps({"passed": True, "pass_reasons": ["ok"], "fail_reasons": []}),
        encoding="utf-8",
    )
    index = tmp_path / "openbreastus_index.json"
    index.write_text(json.dumps({"summary": {"num_cases": 1, "num_files": 1}, "cases": [{"case_id": "c"}], "warnings": []}), encoding="utf-8")
    case_path = tmp_path / "case.h5"
    case_path.write_text("placeholder", encoding="utf-8")
    manifest = tmp_path / "openbreastus_smoke_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "cases": [{"case_id": "c"}],
                "converted_cases": [{"case_id": "c", "path": str(case_path), "has_measured_attenuation": True}],
            }
        ),
        encoding="utf-8",
    )

    result = audit.audit_repo(
        Path(".").resolve(),
        run_dir=run_dir,
        openbreastus_index=index,
        smoke_manifest=manifest,
        require_v01_dod=True,
    )

    assert result["passed"] is True
    dod_check = [check for check in result["checks"] if check["name"] == "v01_dod_evidence"][0]
    assert dod_check["passing_smoke_algorithms"] == ["attenuation_sirt", "straight_sart"]


def test_v01_audit_accepts_quality_evidence_bundle(tmp_path):
    audit = _load_audit_module()
    evidence_root = tmp_path / "runs"
    expected = {
        "openbreastus_quality_20260531T164948Z": (
            "comparison_artifacts/openbreastus_quality_256_4class_5alg_gray",
            ["straight_cgls", "straight_sirt", "straight_sart", "bent_ray_gn", "rwave_adapter"] * 4,
            True,
        ),
        "nbpslice2d_quality_20260531T162341Z": (
            "comparison_artifacts/nbpslice2d_quality_256_4class_5alg_gray",
            ["straight_cgls", "straight_sirt", "straight_sart", "bent_ray_gn", "rwave_adapter"] * 4,
            True,
        ),
        "external_matlab_adapter_quality_20260531T221421Z": (
            "comparison_artifacts/external_matlab_quality_256_4class_gray",
            ["bent_ray_gn", "rwave_adapter"] * 4,
            True,
        ),
        "fwi_kwave_best_grid_reingest_2b8d7a5": (
            "comparison_artifacts/fwi_case_test201_best_grid_cross_algorithm_gray",
            ["straight_cgls", "straight_sirt", "straight_sart", "bent_ray_gn", "rwave_adapter", "fwi_kwave_adapter"],
            True,
        ),
    }
    for run_name, (artifact_stem, algorithms, needs_run_checks) in expected.items():
        run_dir = evidence_root / run_name
        artifact_path = run_dir / f"{artifact_stem}.png"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(b"png")
        summary_path = run_dir / f"{artifact_stem}.summary.csv"
        if algorithms:
            with summary_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["case_id", "algorithm", "rmse", "ssim"])
                writer.writeheader()
                for idx, algorithm in enumerate(algorithms):
                    writer.writerow({"case_id": f"case{idx}", "algorithm": algorithm, "rmse": 1.0, "ssim": 0.9})
        (run_dir / "benchmark_summary.csv").write_text("algorithm,case_id,pass\nx,c,True\n", encoding="utf-8")
        (run_dir / "benchmark_report.md").write_text("# Benchmark report\n", encoding="utf-8")
        if needs_run_checks:
            (run_dir / "benchmark_run_checks.json").write_text(json.dumps({"passed": True}), encoding="utf-8")

    result = audit.audit_repo(Path(".").resolve(), quality_evidence_root=evidence_root)

    assert result["passed"] is True
    evidence_check = [check for check in result["checks"] if check["name"] == "traditional_fwi_quality_evidence"][0]
    assert all(record["passed"] for record in evidence_check["records"])


def test_v01_audit_rejects_incomplete_quality_evidence_bundle(tmp_path):
    audit = _load_audit_module()

    result = audit.audit_repo(Path(".").resolve(), quality_evidence_root=tmp_path / "missing_runs")

    assert result["passed"] is False
    evidence_check = [check for check in result["checks"] if check["name"] == "traditional_fwi_quality_evidence"][0]
    assert evidence_check["fail_reasons"]


def test_v01_audit_rejects_surrogate_attenuation_as_dod_evidence(tmp_path):
    audit = _load_audit_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with (run_dir / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["algorithm", "case_id", "status", "pass", "pass_reasons", "fail_reasons"])
        writer.writeheader()
        writer.writerow({"algorithm": "straight_sart", "case_id": "c", "status": "success", "pass": "True", "pass_reasons": "ok", "fail_reasons": ""})
        writer.writerow({"algorithm": "attenuation_sirt", "case_id": "c", "status": "success", "pass": "True", "pass_reasons": "ok", "fail_reasons": ""})
    _write_report(run_dir)
    _write_run_metadata(run_dir)
    (run_dir / "benchmark_run_checks.json").write_text(
        json.dumps({"passed": True, "pass_reasons": ["ok"], "fail_reasons": []}),
        encoding="utf-8",
    )
    index = tmp_path / "openbreastus_index.json"
    index.write_text(json.dumps({"summary": {"num_cases": 1, "num_files": 1}, "cases": [{"case_id": "c"}], "warnings": []}), encoding="utf-8")
    case_path = tmp_path / "case.h5"
    case_path.write_text("placeholder", encoding="utf-8")
    manifest = tmp_path / "openbreastus_smoke_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "cases": [{"case_id": "c"}],
                "converted_cases": [
                    {
                        "case_id": "c",
                        "path": str(case_path),
                        "has_measured_attenuation": False,
                        "attenuation_evidence": "surrogate_zero_log_amp",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = audit.audit_repo(
        Path(".").resolve(),
        run_dir=run_dir,
        openbreastus_index=index,
        smoke_manifest=manifest,
        require_v01_dod=True,
    )

    assert result["passed"] is False
    dod_check = [check for check in result["checks"] if check["name"] == "v01_dod_evidence"][0]
    assert dod_check["attenuation_capable_case_ids"] == []
    assert "attenuation_sirt has no passing smoke case with measured or simulated nonzero attenuation evidence" in dod_check["fail_reasons"]


def test_v01_audit_run_dir_requires_provenance_artifacts(tmp_path):
    audit = _load_audit_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with (run_dir / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["algorithm", "case_id", "pass", "pass_reasons", "fail_reasons"])
        writer.writeheader()
        writer.writerow({"algorithm": "a", "case_id": "c", "pass": "True", "pass_reasons": "ok", "fail_reasons": ""})
    (run_dir / "benchmark_report.md").write_text("# Benchmark report\n", encoding="utf-8")
    (run_dir / "benchmark_run_checks.json").write_text(json.dumps({"passed": True, "pass_reasons": ["ok"], "fail_reasons": []}), encoding="utf-8")

    result = audit.audit_repo(Path(".").resolve(), run_dir=run_dir)

    assert result["passed"] is False
    run_check = [check for check in result["checks"] if check["name"] == "benchmark_run_artifacts"][0]
    assert str(run_dir / "run_metadata.yaml") in run_check["missing"]


def test_v01_audit_run_dir_requires_report_provenance(tmp_path):
    audit = _load_audit_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with (run_dir / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["algorithm", "case_id", "pass", "pass_reasons", "fail_reasons"])
        writer.writeheader()
        writer.writerow({"algorithm": "a", "case_id": "c", "pass": "True", "pass_reasons": "ok", "fail_reasons": ""})
    (run_dir / "benchmark_report.md").write_text("# Benchmark report\n", encoding="utf-8")
    _write_run_metadata(run_dir)
    (run_dir / "benchmark_run_checks.json").write_text(json.dumps({"passed": True, "pass_reasons": ["ok"], "fail_reasons": []}), encoding="utf-8")

    result = audit.audit_repo(Path(".").resolve(), run_dir=run_dir)

    assert result["passed"] is False
    run_check = [check for check in result["checks"] if check["name"] == "benchmark_run_artifacts"][0]
    assert "Git commit:" in run_check["missing_provenance_fields"]
