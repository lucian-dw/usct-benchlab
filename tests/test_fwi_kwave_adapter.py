from __future__ import annotations

import os
import subprocess

import numpy as np
import pytest

from usctbench.algorithms.fwi.kwave_adapter import KWaveFWIAdapterAlgorithm, read_kwave_fwi_result
from usctbench.data.synthetic import make_sound_speed_case
from usctbench.schema import AlgorithmConfig

h5py = pytest.importorskip("h5py")


def test_read_kwave_fwi_result_extracts_metrics(tmp_path):
    result_path = tmp_path / "result.mat"
    _write_result(result_path)

    result = read_kwave_fwi_result(result_path)

    assert result["sound_speed_mps"].shape == (6, 6)
    assert result["attenuation_np_per_m"].shape == (6, 6)
    assert result["ground_truth_sound_speed_mps"].shape == (6, 6)
    assert result["sound_speed_iter_mps"].shape == (3, 6, 6)
    assert result["iterations"] == 3
    assert result["loss_decreased"] is True
    assert result["dataset_path"] == "/tmp/dataset.mat"


def test_kwave_adapter_ingests_existing_result(tmp_path):
    result_path = tmp_path / "result.mat"
    _write_result(result_path)
    case = make_sound_speed_case(shape=(4, 4), n_transducers=8)

    result = KWaveFWIAdapterAlgorithm().run(case, AlgorithmConfig(parameters={"result_path": str(result_path)}))

    assert result.status == "success"
    assert result.sound_speed_mps.shape == case.grid.shape
    assert result.attenuation_np_per_m.shape == case.grid.shape
    assert result.metrics["external_result_loaded"] is True
    assert result.metrics["iterations"] == 3
    assert result.metrics["selected_iteration"] == 3
    assert result.metrics["loss_decreased"] is True
    assert "rmse" in result.metrics
    assert "kwave_gt_rmse" in result.metrics


def test_kwave_adapter_missing_result_skips(tmp_path):
    case = make_sound_speed_case(shape=(4, 4), n_transducers=8)

    result = KWaveFWIAdapterAlgorithm().run(case, AlgorithmConfig(parameters={"result_path": str(tmp_path / "missing.mat")}))

    assert result.status == "skipped"
    assert "result file not found" in result.failure_reason


def test_kwave_adapter_can_report_fixed_early_iteration(tmp_path):
    result_path = tmp_path / "result.mat"
    _write_result(result_path)
    case = make_sound_speed_case(shape=(4, 4), n_transducers=8)

    result = KWaveFWIAdapterAlgorithm().run(
        case,
        AlgorithmConfig(parameters={"result_path": str(result_path), "reconstruction_iteration": 1}),
    )

    assert result.status == "success"
    assert result.metrics["selected_iteration"] == 1
    assert result.metrics["selected_loss"] == 3.0
    expected = np.linspace(1450.0, 1520.0, 36, dtype=np.float32).reshape(6, 6) - 5.0
    assert np.isclose(float(np.mean(result.sound_speed_mps)), float(np.mean(expected)))


def test_kwave_adapter_expands_env_result_path(tmp_path, monkeypatch):
    result_path = tmp_path / "result.mat"
    _write_result(result_path)
    monkeypatch.setenv("KWAVE_RESULT_FOR_TEST", str(result_path))
    case = make_sound_speed_case(shape=(4, 4), n_transducers=8)

    result = KWaveFWIAdapterAlgorithm().run(case, AlgorithmConfig(parameters={"result_path": "$KWAVE_RESULT_FOR_TEST"}))

    assert result.status == "success"
    assert os.path.samefile(result.metrics["external_result_path"], result_path)


def test_kwave_adapter_full_pipeline_launch_uses_speed_map_command(tmp_path, monkeypatch):
    result_path = tmp_path / "result.mat"
    log_path = tmp_path / "external.log"
    source_mat = tmp_path / "breast_test_speed.mat"
    source_mat.write_text("placeholder", encoding="utf-8")
    dataset_path = tmp_path / "dataset.mat"
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        _write_result(result_path)
        return subprocess.CompletedProcess(command, 0, stdout="ok")

    monkeypatch.setattr("usctbench.algorithms.fwi.kwave_adapter.subprocess.run", fake_run)
    monkeypatch.setenv("KWAVE_PY_FOR_TEST", "/opt/usct-kwave/bin/python")
    monkeypatch.setenv("KWAVE_SOURCE_FOR_TEST", str(source_mat))
    case = make_sound_speed_case(shape=(4, 4), n_transducers=8)

    result = KWaveFWIAdapterAlgorithm().run(
        case,
        AlgorithmConfig(
            parameters={
                "result_path": str(result_path),
                "run_external": True,
                "execution_mode": "full_pipeline_from_speed_map",
                "usct_kwave_root": str(tmp_path),
                "python_bin": "$KWAVE_PY_FOR_TEST",
                "mat_path": "$KWAVE_SOURCE_FOR_TEST",
                "mat_key": "breast_test",
                "sample_index": 1,
                "array_mode": "full128",
                "dataset_path": str(dataset_path),
                "external_log_path": str(log_path),
                "start_matlab": True,
                "overwrite": True,
                "sos_freqs_mhz": [0.3],
                "sos_iters": [20],
            }
        ),
    )

    assert result.status == "success"
    command = commands[0]
    assert command[0] == "/opt/usct-kwave/bin/python"
    assert "--skip-siminfo" not in command
    assert command[command.index("--mat-path") + 1] == str(source_mat)
    assert command[command.index("--dataset-path") + 1] == str(dataset_path)
    assert "--start-matlab" in command
    assert "--overwrite" in command
    assert command[command.index("--sos-iters") + 1] == "20"
    assert result.metrics["external_execution_mode"] == "full_pipeline_from_speed_map"
    assert result.metrics["external_log_path"] == str(log_path)


def test_kwave_adapter_external_dataset_mode_keeps_skip_flags(tmp_path, monkeypatch):
    result_path = tmp_path / "result.mat"
    dataset_path = tmp_path / "dataset.mat"
    dataset_path.write_text("placeholder", encoding="utf-8")
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        _write_result(result_path)
        return subprocess.CompletedProcess(command, 0, stdout="ok")

    monkeypatch.setattr("usctbench.algorithms.fwi.kwave_adapter.subprocess.run", fake_run)
    case = make_sound_speed_case(shape=(4, 4), n_transducers=8)

    result = KWaveFWIAdapterAlgorithm().run(
        case,
        AlgorithmConfig(
            parameters={
                "result_path": str(result_path),
                "run_external": True,
                "dataset_path": str(dataset_path),
                "usct_kwave_root": str(tmp_path),
            }
        ),
    )

    assert result.status == "success"
    command = commands[0]
    assert "--skip-siminfo" in command
    assert "--skip-rf" in command
    assert "--skip-assemble" in command
    assert command[command.index("--dataset-path") + 1] == str(dataset_path)


def test_kwave_adapter_traveltime_warm_start_runs_three_external_steps(tmp_path, monkeypatch):
    result_path = tmp_path / "result.mat"
    dataset_path = tmp_path / "dataset.mat"
    warm_start_path = tmp_path / "warm_start.mat"
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        _write_result(result_path)
        return subprocess.CompletedProcess(command, 0, stdout="ok")

    monkeypatch.setattr("usctbench.algorithms.fwi.kwave_adapter.subprocess.run", fake_run)
    case = make_sound_speed_case(shape=(4, 4), n_transducers=8)

    result = KWaveFWIAdapterAlgorithm().run(
        case,
        AlgorithmConfig(
            parameters={
                "result_path": str(result_path),
                "run_external": True,
                "execution_mode": "full_pipeline_from_speed_map",
                "dataset_path": str(dataset_path),
                "warm_start_builder": "traveltime",
                "warm_start_path": str(warm_start_path),
                "usct_kwave_root": str(tmp_path),
                "mat_path": str(tmp_path / "speed.mat"),
            }
        ),
    )

    assert result.status == "success"
    assert len(commands) == 3
    assert "--skip-inversion" in commands[0]
    assert "openbreastus_diffusion.kwave_dps.make_traveltime_init" in commands[1]
    assert commands[1][commands[1].index("--output-path") + 1] == str(warm_start_path)
    assert "--skip-siminfo" in commands[2]
    assert "--skip-rf" in commands[2]
    assert "--skip-assemble" in commands[2]
    assert commands[2][commands[2].index("--warm-start-result") + 1] == str(warm_start_path)


def _write_result(path):
    sound_speed = np.linspace(1450.0, 1520.0, 36, dtype=np.float32).reshape(6, 6)
    attenuation = np.linspace(0.0, 1.0, 36, dtype=np.float32).reshape(6, 6)
    dataset_path = "/tmp/dataset.mat"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("VEL_ESTIM", data=sound_speed)
        handle.create_dataset("ATTEN_ESTIM", data=attenuation)
        handle.create_dataset("C_INTERP", data=sound_speed + 1.0)
        handle.create_dataset("VEL_ESTIM_ITER", data=np.stack([sound_speed - 5.0, sound_speed - 2.0, sound_speed], axis=0))
        handle.create_dataset("ATTEN_ESTIM_ITER", data=np.stack([attenuation, attenuation, attenuation], axis=0))
        handle.create_dataset("LOSS_ITER", data=np.asarray([[3.0, 2.0, 1.0]], dtype=np.float64))
        handle.create_dataset("psnr_value", data=np.asarray([[20.0]], dtype=np.float64))
        handle.create_dataset("ssim_value", data=np.asarray([[0.5]], dtype=np.float64))
        handle.create_dataset("datasetPath", data=np.asarray([ord(c) for c in dataset_path], dtype=np.uint16)[:, None])
