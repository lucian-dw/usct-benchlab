from __future__ import annotations

import csv

import pytest
import yaml

from usctbench.benchmark.runner import run_benchmark_suite
from usctbench.cli import main
from usctbench.data.synthetic import make_sound_speed_case
from usctbench.io.hdf5 import write_case_hdf5

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
    assert "Runtime total seconds" in report


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
    with (run_root / "benchmark_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["artifacts_complete"] == "True"
    assert rows[0]["peak_memory_mb"]


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
