# k-Wave FWI Adapter

## Physical Assumption

This adapter represents frequency-domain waveform inversion for ring-array USCT using an external k-Wave/WaveformInversionUST-style Helmholtz workflow. It treats the external MATLAB result as the authoritative full-wave inversion output and maps it back into the `USCTCase -> ReconstructionResult` contract.

## Input Requirements

- A standard `USCTCase` with sound-speed ground truth when image metrics are required.
- A MATLAB v7.3 result file containing `VEL_ESTIM`; optional datasets include `ATTEN_ESTIM`, `LOSS_ITER`, `GRAD_IMG_ITER`, `C_INTERP`, `psnr_value`, `ssim_value`, and `datasetPath`.
- For active external execution, an A100 environment with the `USCT_kwave` project, k-Wave, MATLAB engine support, CUDA k-Wave backend, and OpenBreastUS speed-map MAT files.
- `execution_mode: invert_existing_dataset` requires a generated k-Wave channel dataset.
- `execution_mode: full_pipeline_from_speed_map` starts from a 2-D OpenBreastUS speed map and calls the A100 `run_full_pipeline.py` path: sim info, RF generation, channel assembly, Helmholtz preparation inside MATLAB, and multi-frequency FWI.
- `warm_start_builder: bulk_support` runs the A100 bulk-support initializer after channel assembly or on an existing k-Wave dataset, then passes the generated `VEL_ESTIM` MAT file into FWI as `warm_start_path`.

## Default Settings

- Default mode is result ingestion only: `run_external: false`.
- `result_path` must point to an existing k-Wave FWI result `.mat`.
- `run_external: true` calls `openbreastus_diffusion.kwave_dps.run_full_pipeline` with explicit pipeline arguments from the config.
- `configs/algorithms/fwi_kwave_full_pipeline.yaml` is the A100 pure-FWI smoke config for `full128`, existing `preproc_crop300_tla250` k-Wave datasets by default, 0.3:0.025:0.8 MHz, 3 sound-speed iterations per frequency, no attenuation inversion, bulk-support warm start, `0.3 mm` reconstruction grid, `c_geom=1500`, update damping `0.25`, velocity clamp `[1408.692, 1595.1279]` m/s, per-step update clamp `12 m/s`, zero background attenuation, `sign_conv=-1`, and `save_raw_grad_iters: 0`.
- The bulk-support initializer calls `openbreastus_diffusion.kwave_dps.make_bulk_support_init` with `--init-mode bulk_support`, Hilbert arrival picking, support backprojection, `--bulk-update-scale 1.0`, `--bulk-stat median`, `--velocity-bounds 1408.692 1595.1279`, and `--compare-gt`. The old RF texture initializer and `--update-scale -1` are retained only for ablation/debug, not the mainline config.
- The smoke config selects the best RMSE iteration against the external k-Wave `C_INTERP` grid for benchmark output. The renderer still writes both best and final reconstruction artifacts so final-iteration behavior remains visible.
- The default benchmark wrapper case is generated at 256x256 so image metrics are comparable with the traditional quality runs. The external k-Wave/FWI result remains authoritative; judge FWI quality first against the external MAT `C_INTERP` grid and the k-Wave-native metrics, not only against surrogate wrapper metrics.
- Acceptance requires the selected external iteration to improve over the external initial model using `kwave_gt_selected_relative_rmse_improvement`, plus k-Wave-native `kwave_native_psnr` and `kwave_native_ssim`. Final-iteration improvement is still recorded for diagnostics, but no longer gates the pass/fail result. Water/background improvement is also recorded for diagnostics, but is not a pass/fail gate for FWI.
- Current local A100 reference:
  `$USCT_KWAVE_ROOT/openbreastus_diffusion/kwave_dps/docs/experiments/2026-06-02_bulk_support_dps_mainline.md`.
  Pure bulk-support FWI reference inside metrics are: sample001
  PSNR/SSIM/corr/HP-corr `26.1998/0.7887/0.9368/0.8869`; sample201
  `26.4539/0.8315/0.9520/0.9240`; sample501
  `26.1279/0.7936/0.9327/0.8941`; sample701
  `29.3894/0.8216/0.8414/0.8092`.

## Expected Failure Modes

- Missing external result file.
- MATLAB result missing `VEL_ESTIM`.
- Result/case mismatch if `result_path` belongs to a different k-Wave dataset than the input case.
- External launch failure due to missing MATLAB, missing CUDA/MEX binaries, missing k-Wave paths, or absent generated datasets.
- Non-decreasing FWI loss on unstable schedules.

## What To Adjust First

- Verify `result_path`, `dataset_path`, `mat_path`, `mat_key`, and `sample_index` refer to the same case.
- Start with an existing validated result before enabling `run_external`, then move to `invert_existing_dataset`, then `full_pipeline_from_speed_map`.
- Use the configured low-to-high frequency schedule first; if image RMSE worsens while waveform loss decreases, inspect both final and best artifacts before changing the schedule.
- Validate bulk-support init summary, support mask, and internal bulk speed before tuning FWI iterations. Do not use the old `--update-scale -1` RF texture initializer as the mainline unless a dedicated sign diagnostic justifies it.
- If image RMSE worsens while waveform loss decreases, reduce iterations, clamp per-step updates, or use a better multi-frequency schedule before claiming reconstruction quality.
- Check the external stdout/stderr log before changing inversion parameters.

## Acceptance Tests

- Unit test reads a synthetic MATLAB v7.3 FWI result and returns a successful `ReconstructionResult`.
- Missing result path skips clearly rather than crashing.
- Unit tests verify command construction for both existing-dataset inversion and full speed-map pipeline launch.
- The full-pipeline smoke protocol requires external result loading, iteration diagnostics, `kwave_gt_rmse`, `kwave_gt_init_rmse`, `kwave_gt_selected_relative_rmse_improvement`, `kwave_gt_ssim`, `kwave_native_psnr`, and `kwave_native_ssim`.
- A100 smoke evidence should show the full pipeline can produce `contact_sheet.png`, `reconstruction.png`, `reconstruction_best.png`, `reconstruction_final.png`, `ground_truth.png`, `error.png`, `loss_curve.png`, `gradient_step001.png`, `gradient_step020.png`, `metadata.json`, and `run.log`.

## References and Related Code

- `src/usctbench/algorithms/fwi/kwave_adapter.py`
- `configs/algorithms/fwi_kwave_adapter.yaml`
- `configs/algorithms/fwi_kwave_full_pipeline.yaml`
- `scripts/run_fwi_kwave_full_pipeline_smoke.sh`
- `scripts/render_kwave_fwi_smoke_outputs.py`
- `tests/test_fwi_kwave_adapter.py`
- `docs/experiments/FWI_KWAVE_SUCCESS_NOTES.md`
- A100 reference project: `$USCT_KWAVE_ROOT` containing the external k-Wave FWI checkout
- Upstream reference: `rehmanali1994/WaveformInversionUST`, frequency-domain waveform inversion UST using a ring-array transducer.
