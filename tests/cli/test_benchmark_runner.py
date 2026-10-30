"""Public execution, persistence ordering, evaluation isolation and suite paths."""

import json
import h5py
import numpy as np
import pytest
import yaml

from usctbench import run, suite, evaluate
from usctbench.core.io import write_case_hdf5
from usctbench.core.schema import ReconstructionResult


@pytest.mark.parametrize(
    "name", ["straight_cgls", "straight_sirt", "straight_sart", "bent_ray_gn"]
)
def test_native_algorithm_public_run(synthetic_case, name):
    result = run(
        name,
        synthetic_case,
        {"run_controls": {"max_iterations": 1}},
        score=False,
        preview=False,
    )
    assert result.status in {"completed", "budget_exhausted"}
    assert np.isfinite(result.sound_speed_mps).all()
    assert result.metrics["data_relative_residual"] >= 0
    assert "direction_checks" not in result.metrics
    assert "gradient_checks" not in result.metrics


def test_pressure_algorithm_rejects_tof_only_case(synthetic_case):
    result = run("rwave_adapter", synthetic_case, {}, score=False, preview=False)
    assert result.status == "failed"
    assert "TOF-only" in result.failure_reason


@pytest.mark.parametrize("stage", ["evaluation", "preview"])
def test_postprocessing_failure_keeps_numerical_result(
    synthetic_case, tmp_path, monkeypatch, stage
):
    from usctbench.benchmark import runner

    saved = []

    def fail(*args, **kwargs):
        saved.append((tmp_path / "result.h5").read_bytes())
        assert (tmp_path / "index.json").exists()
        with h5py.File(tmp_path / "result.h5") as handle:
            assert np.isfinite(handle["sound_speed_mps"][()]).all()
        raise RuntimeError("postprocessing unavailable")

    monkeypatch.setattr(
        runner,
        (
            "compute_regional_image_metrics"
            if stage == "evaluation"
            else "write_preview_png"
        ),
        fail,
    )
    result = run(
        "straight_cgls",
        synthetic_case,
        {"run_controls": {"max_iterations": 2}},
        tmp_path,
    )
    assert result.status == "budget_exhausted"
    assert saved and saved[-1] == (tmp_path / "result.h5").read_bytes()
    record = json.loads((tmp_path / "run.json").read_text())
    assert record["runtime_s"] > 0
    assert record["stages"]["save"]["status"] == "completed"
    assert record["stages"][stage]["status"] == "failed"
    assert "postprocessing unavailable" in record["stages"][stage]["message"]


def test_runner_supplies_gt_free_shallow_case(synthetic_case, tmp_path, monkeypatch):
    from usctbench.benchmark import runner

    class Algorithm:
        def run(self, case, config):
            assert case.ground_truth.sound_speed_mps is None
            assert case.measurement is synthetic_case.measurement
            assert case.geometry is synthetic_case.geometry
            return ReconstructionResult(
                algorithm="straight_cgls",
                case_id=case.case_id,
                sound_speed_mps=np.full(case.grid.shape, 1500),
            )

    monkeypatch.setattr(runner, "get_algorithm", lambda name: Algorithm())
    result = run("straight_cgls", synthetic_case, {}, tmp_path)
    assert synthetic_case.ground_truth.sound_speed_mps is not None
    assert result.status == "completed"
    assert result.metrics["stages"]["evaluation"]["status"] == "completed"


def test_no_ground_truth_keeps_holdout_without_image_scores(synthetic_case, tmp_path):
    case = synthetic_case.model_copy(deep=True)
    case.ground_truth.sound_speed_mps = None
    result = run(
        "straight_cgls",
        case,
        {
            "run_controls": {"max_iterations": 3},
            "evaluation": {"receiver_fraction": 0.125, "seed": 42},
        },
        tmp_path,
    )
    metrics = result.metrics
    assert metrics["rmse"] is metrics["psnr"] is metrics["ssim"] is None
    assert metrics["image_evaluation"]["status"] == "ground_truth_unavailable"
    assert metrics["evaluation"]["receiver"]["weighted_relative_residual"] >= 0
    assert (
        json.loads((tmp_path / "metrics.json").read_text())["image_evaluation"]
        == metrics["image_evaluation"]
    )


def test_image_region_changes_only_reporting(synthetic_case, tmp_path):
    results = [
        run(
            "straight_cgls",
            synthetic_case,
            {
                "run_controls": {"max_iterations": 3},
                "image_evaluation": {"primary_region": region},
            },
            tmp_path / region,
        )
        for region in ("tissue", "full_image")
    ]
    np.testing.assert_array_equal(
        results[0].sound_speed_mps, results[1].sound_speed_mps
    )
    a, b = [result.metrics for result in results]
    assert a["rmse"] == a["tissue_rmse"]
    assert b["rmse"] == b["full_image_rmse"]
    for row_a, row_b in zip(a["iteration_history"], b["iteration_history"]):
        for key in ("iteration", "residual_norm", "objective"):
            assert row_a[key] == row_b[key]
    assert a["stop_reason"] == b["stop_reason"]


@pytest.mark.parametrize("checkpoint", [False, True])
def test_failures_and_partial_checkpoint_are_not_published_as_success(
    synthetic_case, tmp_path, monkeypatch, checkpoint
):
    from usctbench.benchmark import runner

    class Algorithm:
        def run(self, case, config):
            if not checkpoint:
                raise FloatingPointError("invalid PDE solution")
            return ReconstructionResult(
                algorithm="straight_cgls",
                case_id=case.case_id,
                sound_speed_mps=np.full(case.grid.shape, 1500),
                metrics={"stop_reason": "line_search_failed"},
            )

    monkeypatch.setattr(runner, "get_algorithm", lambda name: Algorithm())
    result = run("straight_cgls", synthetic_case, {}, tmp_path)
    assert result.status == "failed"
    assert (result.sound_speed_mps is not None) == checkpoint
    assert not (tmp_path / "index.json").exists()
    report = evaluate(tmp_path)
    assert not report["passed"]
    assert report["records"][0]["failure_stage"] == "solve"
    assert report["records"][0]["message"]


def test_run_writes_artifacts_and_rejects_config_mismatch(synthetic_case, tmp_path):
    config = tmp_path / "algorithm.yaml"
    config.write_text("name: straight_cgls\nrun_controls:\n  max_iterations: 2\n")
    result = run("straight_cgls", synthetic_case, config, tmp_path / "output")
    assert result.status == "budget_exhausted"
    for name in ("result.h5", "run.json", "index.json", "metrics.json", "preview.png"):
        assert (tmp_path / "output" / name).exists()
    assert evaluate(tmp_path / "output")["passed"]
    result = run("straight_sirt", synthetic_case, config, tmp_path / "output")
    assert result.status == "failed"
    assert "algorithm/config mismatch" in result.failure_reason
    assert not (tmp_path / "output" / "index.json").exists()
    assert not (tmp_path / "output" / "preview.png").exists()
    assert "rmse" not in json.loads((tmp_path / "output" / "metrics.json").read_text())


def test_partial_deployment_override_keeps_explicit_backend():
    from usctbench import prepare_config

    config = prepare_config(
        "fwi_wust",
        {"deployment": {"backend": "cpu", "matlab_executable": "/opt/matlab"}},
        deployment={"runtime_root": "/opt/wust"},
    )
    assert config.deployment.backend == "cpu"
    assert config.deployment.matlab_executable == "/opt/matlab"
    assert config.deployment.runtime_root == "/opt/wust"


def test_suite_resolves_all_paths_relative_to_yaml(
    synthetic_case, tmp_path, monkeypatch
):
    write_case_hdf5(synthetic_case, tmp_path / "cases" / "case.h5")
    (tmp_path / "algorithm.yaml").write_text(
        "name: straight_sirt\nrun_controls:\n  max_iterations: 2\n"
    )
    (tmp_path / "cgls.yaml").write_text(
        "name: straight_cgls\nrun_controls:\n  max_iterations: 2\n"
    )
    path = tmp_path / "suite.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "case_glob": "cases/*.h5",
                "outputs": {"root": "runs"},
                "algorithms": [
                    {"name": "straight_sirt", "config": "algorithm.yaml"},
                    {"name": "straight_cgls", "config": "cgls.yaml"},
                ],
            }
        )
    )
    monkeypatch.chdir(tmp_path.parent)
    from usctbench.benchmark import runner
    from unittest.mock import Mock

    reader = Mock(wraps=runner.read_case_hdf5)
    monkeypatch.setattr(runner, "read_case_hdf5", reader)
    result = suite(path)
    assert result["passed"]
    assert len(result["records"]) == 2
    assert reader.call_count == 1
    assert result["num_cases"] == 1
    assert result["run_root"].startswith(str(tmp_path / "runs"))
    empty = tmp_path / "empty.yaml"
    empty.write_text("case_glob: missing/*.h5\n")
    with pytest.raises(ValueError, match="matched no cases"):
        suite(empty)
