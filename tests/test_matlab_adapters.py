from __future__ import annotations

import yaml

from usctbench.algorithms.adapters.refraction_gn import BentRayGNAdapter
from usctbench.algorithms.adapters.rwave import RWaveAdapter
from usctbench.cli import main
from usctbench.data.synthetic import make_sound_speed_case
from usctbench.io.hdf5 import write_case_hdf5
from usctbench.schema import AlgorithmConfig


def test_matlab_adapters_skip_without_matlab():
    case = make_sound_speed_case(shape=(8, 8), n_transducers=10)
    config = AlgorithmConfig(parameters={"matlab_bin": "/definitely/missing/matlab"})

    for adapter in (BentRayGNAdapter(), RWaveAdapter()):
        result = adapter.run(case, config)
        assert result.status == "skipped"
        assert result.failure_reason


def test_cli_adapter_skip_writes_failure_report(tmp_path):
    case_path = write_case_hdf5(make_sound_speed_case(shape=(8, 8), n_transducers=10), tmp_path / "case.h5")
    config_path = tmp_path / "adapter.yaml"
    config_path.write_text(
        yaml.safe_dump({"parameters": {"matlab_bin": "/definitely/missing/matlab"}}),
        encoding="utf-8",
    )

    exit_code = main(["run", "bent_ray_gn", "--case", str(case_path), "--config", str(config_path), "--out", str(tmp_path / "runs")])

    case_dir = tmp_path / "runs" / "synthetic_circular_sos"
    assert exit_code == 1
    assert (case_dir / "result.h5").exists()
    assert (case_dir / "failure_report.md").exists()
    assert "MATLAB executable not found" in (case_dir / "failure_report.md").read_text(encoding="utf-8")

