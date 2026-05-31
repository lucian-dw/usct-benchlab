from __future__ import annotations

import json

import pytest
import yaml

from usctbench.cli import main
from usctbench.data.synthetic import make_attenuation_case, make_sound_speed_case
from usctbench.io.hdf5 import write_case_hdf5

pytest.importorskip("h5py")


def test_cli_run_writes_standard_artifacts(tmp_path):
    case_path = write_case_hdf5(make_sound_speed_case(shape=(12, 12), n_transducers=16), tmp_path / "case.h5")
    config_path = tmp_path / "straight_cgls.yaml"
    config_path.write_text(
        yaml.safe_dump({"parameters": {"iterations": 8, "reference_sound_speed_mps": 1500.0}}),
        encoding="utf-8",
    )
    out = tmp_path / "runs"

    exit_code = main(["run", "straight_cgls", "--case", str(case_path), "--config", str(config_path), "--out", str(out)])

    case_dir = out / "synthetic_circular_sos"
    assert exit_code == 0
    assert (case_dir / "result.h5").exists()
    assert (case_dir / "metrics.json").exists()
    assert (case_dir / "metadata.yaml").exists()
    assert (case_dir / "preview.png").exists()
    assert (case_dir / "coverage.png").exists()
    assert (case_dir / "coverage_stats.json").exists()
    assert (case_dir / "residual_curve.json").exists()
    metrics = json.loads((case_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "rmse" in metrics
    assert "coverage_nonzero_fraction" in metrics
    assert "ring_artifact_index" in metrics


def test_cli_run_failure_writes_failure_report(tmp_path):
    case_path = write_case_hdf5(make_attenuation_case(shape=(8, 8), n_transducers=10), tmp_path / "case.h5")
    config_path = tmp_path / "straight_cgls.yaml"
    config_path.write_text(yaml.safe_dump({"parameters": {"iterations": 1}}), encoding="utf-8")
    out = tmp_path / "runs"

    exit_code = main(["run", "straight_cgls", "--case", str(case_path), "--config", str(config_path), "--out", str(out)])

    case_dir = out / "synthetic_circular_attenuation"
    assert exit_code == 1
    assert (case_dir / "result.h5").exists()
    assert (case_dir / "failure_report.md").exists()
    assert "delta_tof_s" in (case_dir / "failure_report.md").read_text(encoding="utf-8")


def test_cli_run_fwi_tiny_writes_standard_artifacts(tmp_path):
    case_path = write_case_hdf5(make_sound_speed_case(shape=(10, 10), n_transducers=12), tmp_path / "case.h5")
    config_path = tmp_path / "fwi_tiny.yaml"
    config_path.write_text(
        yaml.safe_dump({"parameters": {"steps": 5, "learning_rate": 1.0e6}}),
        encoding="utf-8",
    )
    out = tmp_path / "runs"

    exit_code = main(["run", "fwi_tiny", "--case", str(case_path), "--config", str(config_path), "--out", str(out)])

    case_dir = out / "synthetic_circular_sos"
    assert exit_code == 0
    assert (case_dir / "result.h5").exists()
    assert (case_dir / "metrics.json").exists()
    assert (case_dir / "metadata.yaml").exists()
    assert (case_dir / "preview.png").exists()
    assert json.loads((case_dir / "metrics.json").read_text(encoding="utf-8"))["loss_decreased"] is True
