from __future__ import annotations

import pytest
import yaml

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
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(yaml.safe_dump({"name": "unit", "thresholds": {"rmse": 2.0}}), encoding="utf-8")

    exit_code = main(["eval", "--run", str(tmp_path / "run"), "--protocol", str(protocol)])

    assert exit_code == 0
    assert (tmp_path / "run" / "benchmark_summary.csv").exists()
    report = (tmp_path / "run" / "benchmark_report.md").read_text(encoding="utf-8")
    assert "Passed: 1" in report


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

