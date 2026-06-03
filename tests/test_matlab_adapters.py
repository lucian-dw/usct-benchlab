from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from usctbench.adapters.matlab import (
    read_matlab_adapter_result,
    write_matlab_adapter_contract,
    write_matlab_adapter_result,
    write_usct_case_mat,
)
from usctbench.algorithms.adapters._matlab_optional import (
    _entrypoint_path,
    _parse_matlab_log_diagnostics,
    requests_matlab_backend,
)
from usctbench.algorithms.adapters.refraction_gn import BentRayGNAdapter
from usctbench.algorithms.adapters.rwave import RWaveAdapter
from usctbench.cli import main
from usctbench.data.synthetic import make_sound_speed_case
from usctbench.io.hdf5 import write_case_hdf5
from usctbench.schema import AlgorithmConfig, MeasurementSpec, ReconstructionResult


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


def test_optional_matlab_backend_accepts_absolute_entrypoint(tmp_path):
    external_root = tmp_path / "external"
    entrypoint = tmp_path / "scripts" / "entrypoint.m"
    external_root.mkdir()
    entrypoint.parent.mkdir()
    entrypoint.write_text("% placeholder", encoding="utf-8")

    assert _entrypoint_path(external_root, str(entrypoint)) == entrypoint
    assert _entrypoint_path(external_root, "relative_entrypoint.m") == external_root / "relative_entrypoint.m"


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


def test_matlab_log_diagnostics_classify_full_greens_sign_jacobian_failure():
    log_text = """
    The number of bad linkings:0
    The number of the rays for which the sign of the Jacobian is changed are:2
    The number of the rays for which the sign of the Jacobian is changed are:9
    The frequency level (linearised subproblem) is:1
    The objective function is:0.0079
    The relative error is:99.87%
    The frequency level (linearised subproblem) is:2
    The objective function is:0.0181
    """

    metrics = _parse_matlab_log_diagnostics(log_text)

    assert metrics["matlab_log_bad_linkings_total"] == 0.0
    assert metrics["matlab_log_sign_jacobian_changed_total"] == 11.0
    assert metrics["matlab_log_sign_jacobian_changed_max"] == 9.0
    assert metrics["matlab_log_objective_increased"] is True
    assert metrics["matlab_log_frequency_level_max"] == 2.0
    assert metrics["matlab_log_relative_error_percent_last"] == 99.87
    assert metrics["matlab_log_likely_failure_mode"] == "sign_jacobian_or_update_direction"


def test_matlab_log_diagnostics_prioritize_nonfinite_gradient():
    metrics = _parse_matlab_log_diagnostics(
        "The objective function is:1.0\n"
        "Warning: NaN gradient detected\n"
        "The number of bad linkings:3\n"
    )

    assert metrics["matlab_log_nan_token_count"] == 1
    assert metrics["matlab_log_bad_linkings_total"] == 3.0
    assert metrics["matlab_log_likely_failure_mode"] == "nan_or_nonfinite_gradient"


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


def test_write_matlab_adapter_contract_writes_helper_functions(tmp_path):
    contract_dir = write_matlab_adapter_contract(tmp_path)

    read_helper = contract_dir / "usctbench_read_case.m"
    write_helper = contract_dir / "usctbench_write_result.m"

    assert read_helper.exists()
    assert write_helper.exists()
    assert "h5read(input_mat, '/measurement/delta_tof_s')" in read_helper.read_text(encoding="utf-8")
    assert "case_data.geometry.radius_m" in read_helper.read_text(encoding="utf-8")
    assert "h5create(output_mat, '/sound_speed_mps'" in write_helper.read_text(encoding="utf-8")


def test_rwave_matlab_entrypoint_is_complex_contract_aware():
    entrypoint = Path("scripts/matlab_adapters/rwave_tof_greens_entrypoint.m")
    text = entrypoint.read_text(encoding="utf-8")

    assert "case_data.rwave.phase_slope_delay_s" in text
    assert "case_data.rwave.complex_quality" in text
    assert "complex_contract_used" in text
    assert "external_greens_full_wavefield" in text


def test_matlab_adapter_exports_input_before_external_execution_skip(tmp_path, monkeypatch):
    class FakeMatlabAdapter:
        def __init__(self, work_dir):
            self.work_dir = work_dir

        def run_batch(self, code, *, log_name="matlab.log", timeout_s=None):
            assert "usctbench_input_mat" in code
            assert "usctbench_output_mat" in code
            assert "usctbench_parameters_json" in code
            assert "run_refraction.m" in code
            assert "addpath" in code
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
    assert "did not write usctbench_output_mat" in result.failure_reason
    assert result.artifacts["external_entrypoint"].endswith("run_refraction.m")
    input_path = result.artifacts["adapter_input_mat"]
    output_path = result.artifacts["adapter_output_mat"]
    assert input_path.endswith("_input.mat")
    assert output_path.endswith("_output.mat")
    assert (tmp_path / "case_run" / "bent_ray_gn_matlab.log").exists()
    assert Path(input_path).exists()
    assert (tmp_path / "case_run" / "usctbench_read_case.m").exists()
    assert (tmp_path / "case_run" / "usctbench_write_result.m").exists()


def test_matlab_adapter_ingests_standard_external_output(tmp_path, monkeypatch):
    output_path = tmp_path / "case_run" / "adapter_output.mat"

    class FakeMatlabAdapter:
        def __init__(self, work_dir):
            self.work_dir = work_dir

        def run_batch(self, code, *, log_name="matlab.log", timeout_s=None):
            log_path = self.work_dir / log_name
            log_path.write_text(code, encoding="utf-8")
            write_matlab_adapter_result(
                ReconstructionResult(
                    algorithm="rwave_adapter",
                    case_id="synthetic_circular_sos",
                    sound_speed_mps=make_sound_speed_case(shape=(8, 8), n_transducers=10).ground_truth.sound_speed_mps,
                    metrics={"external_metric": 1.25},
                    artifacts={"external_artifact": "ok"},
                ),
                output_path,
            )
            return log_path

    def fake_from_config(*, matlab_bin=None, work_dir=None):
        work_dir = tmp_path / "work" if work_dir is None else Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        return FakeMatlabAdapter(work_dir)

    external_root = tmp_path / "external"
    external_root.mkdir()
    (external_root / "run_rwave.m").write_text("% placeholder", encoding="utf-8")
    monkeypatch.setattr("usctbench.algorithms.adapters._matlab_optional.MatlabAdapter.from_config", fake_from_config)

    case = make_sound_speed_case(shape=(8, 8), n_transducers=10)
    result = RWaveAdapter().run(
        case,
        AlgorithmConfig(
            parameters={
                "backend": "matlab",
                "external_root": str(external_root),
                "entrypoint": "run_rwave.m",
                "_run_output_dir": str(tmp_path / "case_run"),
                "adapter_output_path": str(output_path),
            }
        ),
    )

    assert result.status == "success"
    assert result.sound_speed_mps.shape == case.grid.shape
    assert result.metrics["external_metric"] == 1.25
    assert result.metrics["external_adapter_output_loaded"] is True
    assert result.artifacts["external_artifact"] == "ok"
    assert result.artifacts["adapter_output_mat"] == str(output_path)
    assert result.artifacts["adapter_contract_dir"] == str(tmp_path / "case_run")


def test_matlab_adapter_augments_output_with_log_diagnostics(tmp_path, monkeypatch):
    output_path = tmp_path / "case_run" / "adapter_output.mat"

    class FakeMatlabAdapter:
        def __init__(self, work_dir):
            self.work_dir = work_dir

        def run_batch(self, code, *, log_name="matlab.log", timeout_s=None):
            log_path = self.work_dir / log_name
            log_path.write_text(
                "The number of bad linkings:0\n"
                "The number of the rays for which the sign of the Jacobian is changed are:5\n"
                "The objective function is:0.01\n"
                "The objective function is:0.02\n",
                encoding="utf-8",
            )
            write_matlab_adapter_result(
                ReconstructionResult(
                    algorithm="rwave_adapter",
                    case_id="synthetic_circular_sos",
                    sound_speed_mps=make_sound_speed_case(shape=(8, 8), n_transducers=10).ground_truth.sound_speed_mps,
                    metrics={"external_metric": 1.25},
                ),
                output_path,
            )
            return log_path

    def fake_from_config(*, matlab_bin=None, work_dir=None):
        work_dir = tmp_path / "work" if work_dir is None else Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        return FakeMatlabAdapter(work_dir)

    external_root = tmp_path / "external"
    external_root.mkdir()
    (external_root / "run_rwave.m").write_text("% placeholder", encoding="utf-8")
    monkeypatch.setattr("usctbench.algorithms.adapters._matlab_optional.MatlabAdapter.from_config", fake_from_config)

    result = RWaveAdapter().run(
        make_sound_speed_case(shape=(8, 8), n_transducers=10),
        AlgorithmConfig(
            parameters={
                "backend": "matlab",
                "external_root": str(external_root),
                "entrypoint": "run_rwave.m",
                "_run_output_dir": str(tmp_path / "case_run"),
                "adapter_output_path": str(output_path),
            }
        ),
    )

    assert result.status == "success"
    assert result.metrics["external_metric"] == 1.25
    assert result.metrics["matlab_log_sign_jacobian_changed_total"] == 5.0
    assert result.metrics["matlab_log_objective_increased"] is True
    assert result.metrics["matlab_log_likely_failure_mode"] == "sign_jacobian_or_update_direction"


def test_matlab_adapter_augments_external_output_metrics(tmp_path, monkeypatch):
    output_path = tmp_path / "case_run" / "adapter_output.mat"

    class FakeMatlabAdapter:
        def __init__(self, work_dir):
            self.work_dir = work_dir

        def run_batch(self, code, *, log_name="matlab.log", timeout_s=None):
            log_path = self.work_dir / log_name
            log_path.write_text(code, encoding="utf-8")
            case = make_sound_speed_case(shape=(10, 10), n_transducers=12)
            write_matlab_adapter_result(
                ReconstructionResult(
                    algorithm="bent_ray_gn",
                    case_id=case.case_id,
                    sound_speed_mps=case.ground_truth.sound_speed_mps,
                ),
                output_path,
            )
            return log_path

    def fake_from_config(*, matlab_bin=None, work_dir=None):
        work_dir = tmp_path / "work" if work_dir is None else Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        return FakeMatlabAdapter(work_dir)

    external_root = tmp_path / "external"
    external_root.mkdir()
    (external_root / "run_refraction.m").write_text("% placeholder", encoding="utf-8")
    monkeypatch.setattr("usctbench.algorithms.adapters._matlab_optional.MatlabAdapter.from_config", fake_from_config)

    case = make_sound_speed_case(shape=(10, 10), n_transducers=12)
    result = BentRayGNAdapter().run(
        case,
        AlgorithmConfig(
            parameters={
                "backend": "matlab",
                "external_root": str(external_root),
                "entrypoint": "run_refraction.m",
                "_run_output_dir": str(tmp_path / "case_run"),
                "adapter_output_path": str(output_path),
            }
        ),
    )

    assert result.status == "success"
    assert result.metrics["rmse"] == 0.0
    assert result.metrics["water_improved"] is True
    assert result.metrics["data_residual_reduction"] > 0.0


def test_matlab_adapter_result_roundtrip(tmp_path):
    result = ReconstructionResult(
        algorithm="bent_ray_gn",
        case_id="case_a",
        sound_speed_mps=make_sound_speed_case(shape=(6, 6), n_transducers=8).ground_truth.sound_speed_mps,
        metrics={"rmse": 2.0},
    )
    path = write_matlab_adapter_result(result, tmp_path / "out.mat")

    loaded = read_matlab_adapter_result(path, algorithm="fallback", case_id="fallback_case")

    assert loaded.algorithm == "bent_ray_gn"
    assert loaded.case_id == "case_a"
    assert loaded.metrics["rmse"] == 2.0
    assert loaded.sound_speed_mps.shape == (6, 6)


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


def _with_feature_channel(case, channel: str):
    return case.model_copy(update={"metadata": {**case.metadata, "feature_channel": channel}})
