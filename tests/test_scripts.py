from __future__ import annotations

import os
from pathlib import Path
import subprocess


def test_run_smoke_runs_benchmark_when_cases_exist():
    text = Path("scripts/run_smoke.sh").read_text(encoding="utf-8")

    assert "usctbench.cli bench" in text
    assert "audit_v01_readiness.py" in text
    assert "USCT_SAMPLE_ROOT/cases" in text
    assert "USCT_REQUIRE_SMOKE_CASES" in text


def test_shell_scripts_source_local_env():
    for path in [
        Path("scripts/setup_workspace.sh"),
        Path("scripts/check_server.sh"),
        Path("scripts/bootstrap_a100.sh"),
        Path("scripts/run_smoke.sh"),
        Path("scripts/run_v01_release_check.sh"),
        Path("scripts/run_fwi_kwave_adapter_smoke.sh"),
        Path("scripts/run_fwi_kwave_full_pipeline_smoke.sh"),
        Path("scripts/run_nbpslice2d_smoke.sh"),
        Path("scripts/run_openbreastus_quality.sh"),
        Path("scripts/run_nbpslice2d_quality.sh"),
    ]:
        text = path.read_text(encoding="utf-8")
        assert 'if [ -f "$REPO_DIR/.env" ]; then' in text
        assert '. "$REPO_DIR/.env"' in text


def test_v01_release_check_runs_full_evidence_chain():
    text = Path("scripts/run_v01_release_check.sh").read_text(encoding="utf-8")

    assert "pytest -q" in text
    assert "inspect-openbreastus" in text
    assert "make-smoke" in text
    assert "usctbench.cli bench" in text
    assert "--require-v01-dod" in text
    assert "--openbreastus-index" in text
    assert "--smoke-manifest" in text
    assert "bench_status=$?" in text
    assert "audit_status=$?" in text


def test_fwi_kwave_adapter_smoke_script_runs_adapter_flow():
    text = Path("scripts/run_fwi_kwave_adapter_smoke.sh").read_text(encoding="utf-8")

    assert "convert_kwave_channel_mat" in text
    assert "fwi_kwave_adapter" in text
    assert "fwi_kwave_adapter_smoke.yaml" in text
    assert "USCT_KWAVE_FWI_RESULT_PATH" in text


def test_fwi_kwave_adapter_smoke_protocol_is_ingestion_only():
    import yaml

    protocol = yaml.safe_load(Path("configs/benchmarks/fwi_kwave_adapter_smoke.yaml").read_text(encoding="utf-8"))
    required = protocol["required_metrics"]["fwi_kwave_adapter"]

    assert "external_result_loaded" in required
    assert "selected_iteration" in required
    assert "selected_loss" in required
    assert "kwave_native_psnr" in required
    assert "kwave_native_ssim" in required
    assert "loss_decreased" not in required
    assert "loss_decreased" not in protocol["minimums"]["fwi_kwave_adapter"]
    assert "water_relative_rmse_improvement" not in required


def test_fwi_kwave_full_pipeline_smoke_script_runs_speed_map_flow():
    text = Path("scripts/run_fwi_kwave_full_pipeline_smoke.sh").read_text(encoding="utf-8")

    assert "convert_speed_mat_volume" in text
    assert "fwi_kwave_full_pipeline.yaml" in text
    assert "render_kwave_fwi_smoke_outputs.py" in text
    assert "USCT_KWAVE_SOURCE_MAT" in text
    assert "USCT_KWAVE_PYTHON_BIN" in text
    assert "USCT_KWAVE_WARM_START_PATH" in text
    assert "USCT_KWAVE_RECONSTRUCTION_ITERATION" in text
    assert "USCT_KWAVE_SAMPLE_INDEX:-201" in text
    assert "USCT_KWAVE_RUN_ID:-fwi_kwave_full_pipeline_success201_dense" in text
    assert "USCT_KWAVE_WRAPPER_CONVERTED_SHAPE:-256" in text
    assert "USCT_KWAVE_RECONSTRUCTION_ITERATION:-final" in text
    assert "benchmark wrapper" in text
    assert "output_shape=($USCT_KWAVE_WRAPPER_CONVERTED_SHAPE, $USCT_KWAVE_WRAPPER_CONVERTED_SHAPE)" in text
    assert "--render-best-and-final" in text


def test_fwi_kwave_render_script_writes_contact_sheet():
    text = Path("scripts/render_kwave_fwi_smoke_outputs.py").read_text(encoding="utf-8")

    assert "contact_sheet.png" in text
    assert "_write_contact_sheet" in text
    assert "k-Wave FWI comparison" in text
    assert "Ground truth" in text


def test_fwi_kwave_full_pipeline_config_uses_multifrequency_rf_warm_start():
    import yaml

    config = yaml.safe_load(Path("configs/algorithms/fwi_kwave_full_pipeline.yaml").read_text(encoding="utf-8"))
    params = config["parameters"]
    expected_freqs = [
        0.3,
        0.325,
        0.35,
        0.375,
        0.4,
        0.425,
        0.45,
        0.475,
        0.5,
        0.525,
        0.55,
        0.575,
        0.6,
        0.625,
        0.65,
        0.675,
        0.7,
        0.725,
        0.75,
        0.775,
        0.8,
    ]

    assert params["array_mode"] == "full128"
    assert params["object_scale"] == 1.0
    assert params["object_pose"] == "native"
    assert params["background_speed"] == 1500.0
    assert params["ncalc"] == 512
    assert params["xmax_mm"] == 120.0
    assert params["circle_radius_mm"] == 110.0
    assert params["backend"] == "cuda-binary"
    assert params["warm_start_builder"] == "traveltime"
    assert params["start_matlab"] is True
    assert params["no_connect_existing"] is True
    assert params["sos_freqs_mhz"] == expected_freqs
    assert params["sos_iters"] == [3]
    assert params["atten_iters"] == [0]
    assert params["recon_dxi_mm"] == 0.3
    assert params["c_geom"] == 1500.0
    assert params["c_init"] == 1500.0
    assert params["step_damping"] == 0.25
    assert params["max_update_mps"] == 12.0
    assert params["velocity_bounds"] == [1408.692, 1595.1279]
    assert params["exclude_neighbor_fraction"] == 0.123046875
    assert params["perc_outliers"] == 0.99
    assert params["tof_pre_frac"] == 0.05
    assert params["filter_cutoff"] == 0.75
    assert params["atten_bkgnd"] == 0.0
    assert params["sos2atten"] == 0.0
    assert params["save_raw_grad_iters"] == 0
    assert params["reconstruction_iteration"] == "final"
    warm_args = params["warm_start_args"]
    assert warm_args[warm_args.index("--background-speed") + 1] == "1500"
    assert warm_args[warm_args.index("--coarse-dxi-mm") + 1] == "3.0"
    assert warm_args[warm_args.index("--support-mode") + 1] == "backprojection"
    assert warm_args[warm_args.index("--velocity-bounds") + 1 : warm_args.index("--velocity-bounds") + 3] == [
        "1408.7",
        "1595.1",
    ]
    assert warm_args[warm_args.index("--update-scale") + 1] == "-1"
    assert "--compare-gt" in warm_args


def test_fwi_kwave_full_pipeline_protocol_uses_native_kwave_metrics():
    import yaml

    protocol = yaml.safe_load(Path("configs/benchmarks/fwi_kwave_full_pipeline_smoke.yaml").read_text(encoding="utf-8"))
    required = protocol["required_metrics"]["fwi_kwave_adapter"]

    assert "best_iteration" in required
    assert "final_iteration_rmse" in required
    assert "kwave_gt_rmse" in required
    assert "kwave_gt_init_rmse" in required
    assert "kwave_gt_final_relative_rmse_improvement" in required
    assert "kwave_gt_ssim" in required
    assert "kwave_native_psnr" in required
    assert "kwave_native_ssim" in required
    assert "kwave_gt_water_relative_rmse_improvement" not in required
    assert "water_relative_rmse_improvement" not in required
    assert "kwave_gt_water_relative_rmse_improvement" not in protocol["minimums"]["fwi_kwave_adapter"]
    assert "kwave_gt_final_relative_rmse_improvement" in protocol["minimums"]["fwi_kwave_adapter"]
    assert "water_relative_rmse_improvement" not in protocol["minimums"]["fwi_kwave_adapter"]
    assert "kwave_gt_init_water_relative_rmse_improvement" not in protocol["minimums"]["fwi_kwave_adapter"]


def test_nbpslice2d_smoke_script_runs_dataset_flow():
    text = Path("scripts/run_nbpslice2d_smoke.sh").read_text(encoding="utf-8")

    assert "inspect-nbpslice2d" in text
    assert "make-nbp-smoke" in text
    assert "nbpslice2d_smoke.yaml" in text
    assert "USCT_NBP_ZIP_PATH" in text


def test_quality_scripts_render_standard_comparison_panels():
    openbreast = Path("scripts/run_openbreastus_quality.sh").read_text(encoding="utf-8")
    nbp = Path("scripts/run_nbpslice2d_quality.sh").read_text(encoding="utf-8")

    for text in (openbreast, nbp):
        assert "render_class_comparison_panels.py" in text
        assert "comparison_artifacts" in text
        assert "--field sound_speed" in text
        assert "--cmap gray" in text
        assert "straight_cgls straight_sirt straight_sart bent_ray_gn rwave_adapter" in text
    assert "openbreastus_quality_256_sound_speed_gray.png" in openbreast
    assert "nbpslice2d_quality_256_sound_speed_gray.png" in nbp


def test_setup_workspace_root_layout_symlinks_resolve_to_workspace_dirs(tmp_path):
    repo = tmp_path / "usct-benchlab"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    script = scripts / "setup_workspace.sh"
    script.write_text(Path("scripts/setup_workspace.sh").read_text(encoding="utf-8"), encoding="utf-8")
    script.chmod(0o755)

    subprocess.run(["bash", str(script)], cwd=repo, check=True, env={**os.environ, "USCT_WORKSPACE": str(repo)})

    assert (repo / "data" / "raw" / "openbreastus").resolve() == repo / "data" / "openbreastus"
    assert (repo / "data" / "processed" / "openbreastus_sample").resolve() == repo / "data" / "openbreastus_sample"
    assert (repo / "runs" / "current").resolve() == repo / "runs" / "usctbench_runs"
    assert (repo / "external" / "local_external").resolve() == repo / "external"
    assert (repo / "checkpoints" / "local_checkpoints").resolve() == repo / "checkpoints"


def test_setup_workspace_split_layout_symlinks_resolve_to_workspace_dirs(tmp_path):
    workspace = tmp_path / "workspace"
    repo = workspace / "code"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    script = scripts / "setup_workspace.sh"
    script.write_text(Path("scripts/setup_workspace.sh").read_text(encoding="utf-8"), encoding="utf-8")
    script.chmod(0o755)

    subprocess.run(["bash", str(script)], cwd=repo, check=True, env={**os.environ, "USCT_WORKSPACE": str(workspace)})

    assert (repo / "data" / "raw" / "openbreastus").resolve() == workspace / "data" / "openbreastus"
    assert (repo / "data" / "processed" / "openbreastus_sample").resolve() == workspace / "data" / "openbreastus_sample"
    assert (repo / "runs" / "current").resolve() == workspace / "runs" / "usctbench_runs"
    assert (repo / "external" / "local_external").resolve() == workspace / "external"
    assert (repo / "checkpoints" / "local_checkpoints").resolve() == workspace / "checkpoints"
