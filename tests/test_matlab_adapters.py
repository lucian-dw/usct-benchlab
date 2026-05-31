from __future__ import annotations

from pathlib import Path

import yaml

from usctbench.adapters.matlab import write_usct_case_mat
from usctbench.algorithms.adapters._matlab_optional import requests_matlab_backend
from usctbench.algorithms.adapters.refraction_gn import BentRayGNAdapter
from usctbench.algorithms.adapters.rwave import RWaveAdapter
from usctbench.cli import main
from usctbench.data.synthetic import make_sound_speed_case
from usctbench.io.hdf5 import write_case_hdf5
from usctbench.schema import AlgorithmConfig


def test_adapter_native_backends_reconstruct_sound_speed():
    case = make_sound_speed_case(shape=(12, 12), n_transducers=16)
    config = AlgorithmConfig(
        parameters={
            "backend": "python",
            "outer_iterations": 2,
            "inner_iterations": 6,
            "regularization": "laplacian",
            "regularization_lambda": 1.0e-5,
            "smooth_sigma": 0.0,
        }
    )

    for adapter in (BentRayGNAdapter(), RWaveAdapter()):
        result = adapter.run(case, config)
        assert result.status == "success"
        assert result.sound_speed_mps.shape == case.grid.shape
        assert result.metrics["data_residual_reduction"] > 0.0
        assert result.metrics["rmse"] < 100.0
        assert result.metrics["method_family"]


def test_optional_matlab_backend_request_detection():
    assert requests_matlab_backend(AlgorithmConfig(parameters={"backend": "python"})) is False
    assert requests_matlab_backend(AlgorithmConfig(parameters={"backend": "matlab"})) is True
    assert requests_matlab_backend(AlgorithmConfig(parameters={"external_root": "/tmp/external"})) is True
    assert requests_matlab_backend(AlgorithmConfig(parameters={"entrypoint": "run.m"})) is True


def test_matlab_adapters_skip_without_matlab():
    case = make_sound_speed_case(shape=(8, 8), n_transducers=10)
    config = AlgorithmConfig(parameters={"matlab_bin": "/definitely/missing/matlab"})

    for adapter in (BentRayGNAdapter(), RWaveAdapter()):
        result = adapter.run(case, config)
        assert result.status == "skipped"
        assert result.failure_reason
        assert result.metrics["adapter_dependency_available"] is False
        assert result.artifacts["adapter_status"] == "skipped"
        assert result.artifacts["skip_reason"] == result.failure_reason


def test_write_usct_case_mat_exports_standard_adapter_input(tmp_path):
    import h5py

    case = make_sound_speed_case(shape=(8, 8), n_transducers=10)
    path = write_usct_case_mat(case, tmp_path / "adapter_input.mat")

    with h5py.File(path, "r") as handle:
        assert handle.attrs["record_type"] == "USCTCase"
        assert handle.attrs["case_id"] == case.case_id
        assert tuple(handle["grid/shape"][()]) == case.grid.shape
        assert handle["geometry/tx_pos_m"].shape == case.geometry.tx_pos_m.shape
        assert handle["measurement/delta_tof_s"].shape == case.measurement.delta_tof_s.shape
        assert handle["measurement/valid_mask"].dtype.kind in {"u", "i"}
        assert handle["ground_truth/sound_speed_mps"].shape == case.grid.shape


def test_matlab_adapter_exports_input_before_external_execution_skip(tmp_path, monkeypatch):
    class FakeMatlabAdapter:
        def __init__(self, work_dir):
            self.work_dir = work_dir

        def run_batch(self, code, *, log_name="matlab.log", timeout_s=None):
            assert "usctbench_input_mat" in code
            log_path = self.work_dir / log_name
            log_path.write_text(code, encoding="utf-8")
            return log_path

    def fake_from_config(*, matlab_bin=None, work_dir=None):
        work_dir = tmp_path / "work" if work_dir is None else Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        return FakeMatlabAdapter(work_dir)

    external_root = tmp_path / "external"
    external_root.mkdir()
    (external_root / "run_refraction.m").write_text("% placeholder", encoding="utf-8")
    monkeypatch.setattr("usctbench.algorithms.adapters._matlab_optional.MatlabAdapter.from_config", fake_from_config)

    case = make_sound_speed_case(shape=(8, 8), n_transducers=10)
    result = BentRayGNAdapter().run(
        case,
        AlgorithmConfig(
            parameters={
                "backend": "matlab",
                "external_root": str(external_root),
                "entrypoint": "run_refraction.m",
                "_run_output_dir": str(tmp_path / "case_run"),
            }
        ),
    )

    assert result.status == "skipped"
    assert "adapter input was exported" in result.failure_reason
    assert "output ingest is not implemented yet" in result.failure_reason
    assert result.artifacts["external_entrypoint"].endswith("run_refraction.m")
    input_path = result.artifacts["adapter_input_mat"]
    assert input_path.endswith("_input.mat")
    assert (tmp_path / "case_run" / "bent_ray_gn_matlab.log").exists()
    assert Path(input_path).exists()


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
    report = (case_dir / "failure_report.md").read_text(encoding="utf-8")
    assert "MATLAB executable not found" in report
    assert "- Error type: external-dependency" in report
    metadata = yaml.safe_load((case_dir / "metadata.yaml").read_text(encoding="utf-8"))
    assert metadata["error_type"] == "external-dependency"
