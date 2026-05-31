from __future__ import annotations

import csv
import json

import pytest
import yaml

from usctbench.benchmark.runner import run_algorithm_case, run_benchmark_suite
from usctbench.cli import main
from usctbench.data.synthetic import make_sound_speed_case
from usctbench.io.hdf5 import write_case_hdf5
from usctbench.registry import register_algorithm
from usctbench.schema import ReconstructionResult

pytest.importorskip("h5py")


def test_eval_aggregates_run_metrics(tmp_path):
    run_case = tmp_path / "run" / "straight_cgls" / "case001"
    run_case.mkdir(parents=True)
    (run_case / "metrics.json").write_text('{"rmse": 1.5}', encoding="utf-8")
    (run_case / "metadata.yaml").write_text(
        yaml.safe_dump({"algorithm": "straight_cgls", "case_id": "case001", "status": "success", "runtime_s": 0.1}),
        encoding="utf-8",
    )
    (run_case / "result.h5").write_text("placeholder", encoding="utf-8")
    (run_case / "preview.png").write_bytes(b"placeholder")
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(yaml.safe_dump({"name": "unit", "thresholds": {"rmse": 2.0}}), encoding="utf-8")

    exit_code = main(["eval", "--run", str(tmp_path / "run"), "--protocol", str(protocol)])

    assert exit_code == 0
    assert (tmp_path / "run" / "benchmark_summary.csv").exists()
    with (tmp_path / "run" / "benchmark_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["pass"] == "True"
    assert rows[0]["fail_reasons"] == ""
    assert "status is success" in rows[0]["pass_reasons"]
    report = (tmp_path / "run" / "benchmark_report.md").read_text(encoding="utf-8")
    assert "Passed: 1" in report
    assert "Run checks: passed" in report
    assert "Runtime total seconds" in report
    assert (tmp_path / "run" / "benchmark_run_checks.json").exists()


def test_eval_records_fail_reasons_and_failure_report_presence(tmp_path):
    run_case = tmp_path / "run" / "adapter" / "case001"
    run_case.mkdir(parents=True)
    (run_case / "metrics.json").write_text('{"rmse": 5.0}', encoding="utf-8")
    (run_case / "metadata.yaml").write_text(
        yaml.safe_dump(
            {
                "algorithm": "adapter",
                "case_id": "case001",
                "status": "skipped",
                "runtime_s": 0.0,
                "peak_memory_mb": 12.0,
                "failure_reason": "missing external dependency",
            }
        ),
        encoding="utf-8",
    )
    (run_case / "result.h5").write_text("placeholder", encoding="utf-8")
    (run_case / "failure_report.md").write_text("# Failure report\n", encoding="utf-8")
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(yaml.safe_dump({"name": "unit", "thresholds": {"rmse": 2.0}}), encoding="utf-8")

    exit_code = main(["eval", "--run", str(tmp_path / "run"), "--protocol", str(protocol)])

    assert exit_code == 1
    with (tmp_path / "run" / "benchmark_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["pass"] == "False"
    assert rows[0]["failure_report_present"] == "True"
    assert "status is skipped" in rows[0]["fail_reasons"]
    assert "rmse=5 exceeds max 2" in rows[0]["fail_reasons"]


def test_eval_fails_missing_configured_metric(tmp_path):
    run_case = tmp_path / "run" / "straight_cgls" / "case001"
    run_case.mkdir(parents=True)
    (run_case / "metrics.json").write_text('{"data_residual_norm": 0.0}', encoding="utf-8")
    (run_case / "metadata.yaml").write_text(
        yaml.safe_dump({"algorithm": "straight_cgls", "case_id": "case001", "status": "success", "runtime_s": 0.1}),
        encoding="utf-8",
    )
    (run_case / "result.h5").write_text("placeholder", encoding="utf-8")
    (run_case / "preview.png").write_bytes(b"placeholder")
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(yaml.safe_dump({"name": "unit", "thresholds": {"rmse": 2.0}}), encoding="utf-8")

    exit_code = main(["eval", "--run", str(tmp_path / "run"), "--protocol", str(protocol)])

    assert exit_code == 1
    with (tmp_path / "run" / "benchmark_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["pass"] == "False"
    assert "missing threshold metric rmse" in rows[0]["fail_reasons"]


def test_eval_uses_algorithm_specific_metric_limits(tmp_path):
    run_case = tmp_path / "run" / "attenuation_sirt" / "case001"
    run_case.mkdir(parents=True)
    (run_case / "metrics.json").write_text('{"data_residual_norm": 0.0}', encoding="utf-8")
    (run_case / "metadata.yaml").write_text(
        yaml.safe_dump({"algorithm": "attenuation_sirt", "case_id": "case001", "status": "success", "runtime_s": 0.1}),
        encoding="utf-8",
    )
    (run_case / "result.h5").write_text("placeholder", encoding="utf-8")
    (run_case / "preview.png").write_bytes(b"placeholder")
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(
        yaml.safe_dump(
            {
                "name": "unit",
                "required_metrics": {"attenuation_sirt": ["data_residual_norm"]},
                "thresholds": {"attenuation_sirt": {"data_residual_norm": 1.0}},
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["eval", "--run", str(tmp_path / "run"), "--protocol", str(protocol)])

    assert exit_code == 0
    with (tmp_path / "run" / "benchmark_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["pass"] == "True"
    assert "missing threshold metric rmse" not in rows[0]["fail_reasons"]


def test_eval_fails_missing_expected_algorithm_record(tmp_path):
    run_case = tmp_path / "run" / "straight_sart" / "case001"
    run_case.mkdir(parents=True)
    (run_case / "metrics.json").write_text('{"data_residual_norm": 0.0}', encoding="utf-8")
    (run_case / "metadata.yaml").write_text(
        yaml.safe_dump({"algorithm": "straight_sart", "case_id": "case001", "status": "success", "runtime_s": 0.1}),
        encoding="utf-8",
    )
    (run_case / "result.h5").write_text("placeholder", encoding="utf-8")
    (run_case / "preview.png").write_bytes(b"placeholder")
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(
        yaml.safe_dump(
            {
                "name": "unit",
                "algorithms": [
                    {"name": "straight_sart", "config": "unused.yaml"},
                    {"name": "attenuation_sirt", "config": "unused.yaml"},
                ],
                "min_records": 2,
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["eval", "--run", str(tmp_path / "run"), "--protocol", str(protocol)])

    assert exit_code == 1
    report = (tmp_path / "run" / "benchmark_report.md").read_text(encoding="utf-8")
    assert "Run checks: failed" in report
    assert "missing algorithms: attenuation_sirt" in report


def test_bench_runs_suite_on_synthetic_case(tmp_path):
    case_dir = tmp_path / "cases"
    case_dir.mkdir()
    write_case_hdf5(make_sound_speed_case(shape=(10, 10), n_transducers=12), case_dir / "case.h5")

    config = tmp_path / "straight_cgls.yaml"
    config.write_text(
        yaml.safe_dump({"parameters": {"iterations": 4, "reference_sound_speed_mps": 1500.0}}),
        encoding="utf-8",
    )
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        yaml.safe_dump(
            {
                "name": "unit_suite",
                "run_id": "unit_run",
                "case_glob": str(case_dir / "*.h5"),
                "algorithms": [{"name": "straight_cgls", "config": str(config)}],
                "outputs": {"root": str(tmp_path / "runs")},
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["bench", "--suite", str(suite)])

    assert exit_code == 0
    run_root = tmp_path / "runs" / "unit_run"
    assert (run_root / "straight_cgls" / "synthetic_circular_sos" / "result.h5").exists()
    assert (run_root / "benchmark_summary.csv").exists()
    assert (run_root / "run_metadata.yaml").exists()
    with (run_root / "benchmark_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["artifacts_complete"] == "True"
    assert rows[0]["peak_memory_mb"]
    report = (run_root / "benchmark_report.md").read_text(encoding="utf-8")
    assert "Git commit:" in report
    assert "USCT_DATA_ROOT:" in report


def test_cli_make_synthetic_smoke_writes_cases_and_manifest(tmp_path):
    out = tmp_path / "synthetic_smoke"

    exit_code = main(["data", "make-synthetic-smoke", "--out", str(out), "--shape", "12", "--n-transducers", "12"])

    assert exit_code == 0
    assert (out / "manifest.json").exists()
    cases = sorted((out / "cases").glob("*.h5"))
    assert [path.stem for path in cases] == ["synthetic_circular_sos", "synthetic_homogeneous_sos"]


def test_cli_exposes_quality_subset_commands():
    with pytest.raises(SystemExit) as openbreast_help:
        main(["data", "make-quality", "--help"])
    with pytest.raises(SystemExit) as nbp_help:
        main(["data", "make-nbp-quality", "--help"])
    assert openbreast_help.value.code == 0
    assert nbp_help.value.code == 0


def test_bench_suite_rejects_empty_case_glob(tmp_path):
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        yaml.safe_dump(
            {
                "name": "empty_suite",
                "case_glob": str(tmp_path / "missing" / "*.h5"),
                "algorithms": [{"name": "straight_cgls", "config": "configs/algorithms/straight_cgls.yaml"}],
                "outputs": {"root": str(tmp_path / "runs")},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="matched no cases"):
        run_benchmark_suite(suite)


def test_run_algorithm_case_classifies_data_failure(tmp_path):
    config = tmp_path / "straight_cgls.yaml"
    config.write_text(yaml.safe_dump({"parameters": {"iterations": 1}}), encoding="utf-8")

    out_dir = run_algorithm_case("straight_cgls", tmp_path / "missing_case.h5", config, tmp_path / "run")

    metadata = yaml.safe_load((out_dir / "metadata.yaml").read_text(encoding="utf-8"))
    report = (out_dir / "failure_report.md").read_text(encoding="utf-8")
    assert metadata["status"] == "failed"
    assert metadata["error_type"] == "data"
    assert "- Error type: data" in report


def test_run_algorithm_case_writes_failure_report_when_artifact_write_fails(tmp_path):
    class BadMetricAlgorithm:
        name = "bad_metric"

        def run(self, case, config):
            return ReconstructionResult(
                algorithm=self.name,
                case_id=case.case_id,
                sound_speed_mps=case.ground_truth.sound_speed_mps,
                metrics={"not_json": object()},
            )

    register_algorithm("bad_metric", lambda: BadMetricAlgorithm(), replace=True)
    case_path = tmp_path / "case.h5"
    write_case_hdf5(make_sound_speed_case(shape=(8, 8), n_transducers=10), case_path)
    config = tmp_path / "bad_metric.yaml"
    config.write_text(yaml.safe_dump({"parameters": {}}), encoding="utf-8")

    out_dir = run_algorithm_case("bad_metric", case_path, config, tmp_path / "run")

    metadata = yaml.safe_load((out_dir / "metadata.yaml").read_text(encoding="utf-8"))
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    report = (out_dir / "failure_report.md").read_text(encoding="utf-8")
    assert metadata["status"] == "failed"
    assert metadata["error_type"] == "unknown"
    assert "artifact write failed" in metadata["failure_reason"]
    assert metrics["artifact_write_failed"] is True
    assert "- Error type: unknown" in report


def test_eval_rejects_zero_attenuation_signal_when_required(tmp_path):
    run_case = tmp_path / "run" / "attenuation_sirt" / "case001"
    run_case.mkdir(parents=True)
    (run_case / "metrics.json").write_text(
        json.dumps({"data_residual_norm": 0.0, "attenuation_input_signal_norm": 0.0}),
        encoding="utf-8",
    )
    (run_case / "metadata.yaml").write_text(
        yaml.safe_dump({"algorithm": "attenuation_sirt", "case_id": "case001", "status": "success", "runtime_s": 0.1}),
        encoding="utf-8",
    )
    (run_case / "result.h5").write_text("placeholder", encoding="utf-8")
    (run_case / "preview.png").write_bytes(b"placeholder")
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(
        yaml.safe_dump({"minimums": {"attenuation_sirt": {"attenuation_input_signal_norm": 1.0e-12}}}),
        encoding="utf-8",
    )

    exit_code = main(["eval", "--run", str(tmp_path / "run"), "--protocol", str(protocol)])

    assert exit_code == 1
    with (tmp_path / "run" / "benchmark_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert "attenuation_input_signal_norm=0 below min 1e-12" in rows[0]["fail_reasons"]
