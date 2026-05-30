from __future__ import annotations

import os

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
    assert result.metrics["loss_decreased"] is True
    assert "rmse" in result.metrics


def test_kwave_adapter_missing_result_skips(tmp_path):
    case = make_sound_speed_case(shape=(4, 4), n_transducers=8)

    result = KWaveFWIAdapterAlgorithm().run(case, AlgorithmConfig(parameters={"result_path": str(tmp_path / "missing.mat")}))

    assert result.status == "skipped"
    assert "result file not found" in result.failure_reason


def test_kwave_adapter_expands_env_result_path(tmp_path, monkeypatch):
    result_path = tmp_path / "result.mat"
    _write_result(result_path)
    monkeypatch.setenv("KWAVE_RESULT_FOR_TEST", str(result_path))
    case = make_sound_speed_case(shape=(4, 4), n_transducers=8)

    result = KWaveFWIAdapterAlgorithm().run(case, AlgorithmConfig(parameters={"result_path": "$KWAVE_RESULT_FOR_TEST"}))

    assert result.status == "success"
    assert os.path.samefile(result.metrics["external_result_path"], result_path)


def _write_result(path):
    sound_speed = np.linspace(1450.0, 1520.0, 36, dtype=np.float32).reshape(6, 6)
    attenuation = np.linspace(0.0, 1.0, 36, dtype=np.float32).reshape(6, 6)
    dataset_path = "/tmp/dataset.mat"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("VEL_ESTIM", data=sound_speed)
        handle.create_dataset("ATTEN_ESTIM", data=attenuation)
        handle.create_dataset("LOSS_ITER", data=np.asarray([[3.0, 2.0, 1.0]], dtype=np.float64))
        handle.create_dataset("psnr_value", data=np.asarray([[20.0]], dtype=np.float64))
        handle.create_dataset("ssim_value", data=np.asarray([[0.5]], dtype=np.float64))
        handle.create_dataset("datasetPath", data=np.asarray([ord(c) for c in dataset_path], dtype=np.uint16)[:, None])
