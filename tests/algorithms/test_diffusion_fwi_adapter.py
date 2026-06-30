from __future__ import annotations

import json

import numpy as np
from scipy.io import savemat

from usctbench.algorithms.fwi.diffusion_adapter import (
    DiffusionKWaveFWIAdapterAlgorithm,
    read_diffusion_kwave_fwi_result,
)
from usctbench.core.schema import AlgorithmConfig, ResultStatus


def test_diffusion_adapter_ingests_dps_result_file(synthetic_case, tmp_path):
    result_path = tmp_path / "dps_result.mat"
    summary_path = tmp_path / "dps_result.json"
    dataset_path = tmp_path / "dataset.mat"
    checkpoint_path = tmp_path / "checkpoint.pth"
    summary = {
        "dataset_path": str(dataset_path),
        "checkpoint": str(checkpoint_path),
        "steps": 12,
        "freqs_mhz": [0.3, 0.35, 0.4, 0.45],
        "output_selection": "final",
        "reference_hparams": {
            "prior_mode": "score_reg",
            "score_reg_t": 0.10,
            "score_reg_lambda": 0.1,
            "prior_strength": 1.0,
            "prior_mask_mode": "none",
            "physics_position": "pre",
            "physics_inner_steps": 1,
        },
        "selected_metrics": {"psnr": 24.0, "ssim": 0.72},
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    savemat(
        result_path,
        {
            "VEL_DPS_PHYS": np.full((8, 8), 1510.0, dtype=np.float32),
            "VEL_INIT_VIEW": np.full((8, 8), 1500.0, dtype=np.float32),
            "GT_VIEW": np.full((8, 8), 1505.0, dtype=np.float32),
        },
    )

    external = read_diffusion_kwave_fwi_result(result_path, summary_path=summary_path)
    result = DiffusionKWaveFWIAdapterAlgorithm().run(
        synthetic_case,
        AlgorithmConfig(
            parameters={
                "result_path": str(result_path),
                "summary_path": str(summary_path),
                "checkpoint": str(checkpoint_path),
                "dataset_path": str(dataset_path),
                "array_mode": "sparse64",
                "steps": 12,
                "freqs_mhz": [0.3, 0.35, 0.4, 0.45],
                "score_reg_t": 0.10,
                "score_reg_lambda": 0.1,
            }
        ),
    )

    assert external["selected_field"] == "VEL_DPS_PHYS"
    assert result.status == ResultStatus.SUCCESS
    assert result.sound_speed_mps is not None
    assert result.sound_speed_mps.shape == synthetic_case.grid.shape
    assert result.metrics["external_result_loaded"] is True
    assert result.metrics["external_execution_mode"] == "loader_only"
    assert result.metrics["checkpoint"] == str(checkpoint_path)
    assert result.metrics["dataset_path"] == str(dataset_path)
    assert result.metrics["summary_path"] == str(summary_path)
    assert result.metrics["score_reg_t"] == 0.10
    assert result.metrics["score_reg_lambda"] == 0.1
    assert result.metrics["steps"] == 12
    assert result.metrics["freqs_mhz"] == [0.3, 0.35, 0.4, 0.45]
    assert result.metrics["array_mode"] == "sparse64"
    assert result.metrics["selected_step"] == 12
    assert result.metrics["prior_mode"] == "score_reg"
    assert result.metrics["selected_field"] == "VEL_DPS_PHYS"
    assert result.metrics["dps_selected_psnr"] == 24.0
    assert result.artifacts["external_result_path"] == str(result_path)
    assert result.artifacts["external_summary_path"] == str(summary_path)


def test_diffusion_adapter_accepts_view_field_fallback(synthetic_case, tmp_path):
    result_path = tmp_path / "dps_view_only.mat"
    savemat(result_path, {"VEL_DPS_VIEW": np.full((6, 6), 1495.0)})

    result = DiffusionKWaveFWIAdapterAlgorithm().run(
        synthetic_case, AlgorithmConfig(parameters={"result_path": str(result_path)})
    )

    assert result.status == ResultStatus.SUCCESS
    assert result.sound_speed_mps is not None
    assert result.metrics["selected_field"] == "VEL_DPS_VIEW"
