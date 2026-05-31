# k-Wave FWI Adapter

## Physical Assumption

This adapter represents frequency-domain waveform inversion for ring-array USCT using an external k-Wave/WaveformInversionUST-style Helmholtz workflow. It treats the external MATLAB result as the authoritative full-wave inversion output and maps it back into the `USCTCase -> ReconstructionResult` contract.

## Input Requirements

- A standard `USCTCase` with sound-speed ground truth when image metrics are required.
- A MATLAB v7.3 result file containing `VEL_ESTIM`; optional datasets include `ATTEN_ESTIM`, `LOSS_ITER`, `GRAD_IMG_ITER`, `C_INTERP`, `psnr_value`, `ssim_value`, and `datasetPath`.
- For active external execution, an A100 environment with the `USCT_kwave` project, k-Wave, MATLAB engine support, CUDA k-Wave backend, and OpenBreastUS speed-map MAT files.
- `execution_mode: invert_existing_dataset` requires a generated k-Wave channel dataset.
- `execution_mode: full_pipeline_from_speed_map` starts from a 2-D OpenBreastUS speed map and calls the A100 `run_full_pipeline.py` path: sim info, RF generation, channel assembly, Helmholtz preparation inside MATLAB, and multi-frequency FWI.
- `warm_start_builder: traveltime` runs the A100 RF travel-time initializer after channel assembly and passes the generated `VEL_ESTIM` MAT file into FWI as `warm_start_path`.

## Default Settings

- Default mode is result ingestion only: `run_external: false`.
- `result_path` must point to an existing k-Wave FWI result `.mat`.
- `run_external: true` calls `openbreastus_diffusion.kwave_dps.run_full_pipeline` with explicit pipeline arguments from the config.
- `configs/algorithms/fwi_kwave_full_pipeline.yaml` is the A100 smoke config for `full128`, 0.3:0.025:0.8 MHz, 3 sound-speed iterations per frequency, no attenuation inversion, RF travel-time warm start, `0.3 mm` reconstruction grid, `c_geom=1500`, update damping `0.25`, velocity clamp `[1408.692, 1595.1279]` m/s, per-step update clamp `12 m/s`, zero background attenuation, and `save_raw_grad_iters: 0`.
- The RF travel-time initializer pins the successful test201 parameters from `rfinit_densefreq_test201_success.json`, including `--background-speed 1500`, `--recon-dxi-mm 0.3`, support backprojection settings, `--velocity-bounds 1408.7 1595.1`, `--update-scale -1`, and `--compare-gt`.
- The smoke config selects the final iteration for benchmark judgment. The renderer still writes final and best reconstruction artifacts, but best is diagnostic only and should not be used as oracle selection.
- The default benchmark wrapper case is generated at 256x256 so image metrics are comparable with the traditional quality runs. The external k-Wave/FWI result remains authoritative; judge FWI quality first against the external MAT `C_INTERP` grid and the k-Wave-native metrics, not only against surrogate wrapper metrics.
- Current success reference: OpenBreastUS test201, result `/home/wudalong/USCT_kwave/openbreastus_diffusion/kwave_dps/outputs/rfinit_densefreq_test201_20260531_215043/results/test201_rfinit_sos0p3to0p8_step0p025_iter3.mat`, with `final_inside_psnr=22.7387`, `final_inside_corr=0.8933`, and `final_inside_hp_corr=0.8295` in `/home/wudalong/USCT_kwave/openbreastus_diffusion/kwave_dps/configs/rfinit_densefreq_test201_success.json`.

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
- Validate RF warm-start sign, scale, and support before tuning FWI iterations; wrong sign can produce plausible-looking outlines with negative object correlation.
- If image RMSE worsens while waveform loss decreases, reduce iterations, clamp per-step updates, or use a better multi-frequency schedule before claiming reconstruction quality.
- Check the external stdout/stderr log before changing inversion parameters.

## Acceptance Tests

- Unit test reads a synthetic MATLAB v7.3 FWI result and returns a successful `ReconstructionResult`.
- Missing result path skips clearly rather than crashing.
- Unit tests verify command construction for both existing-dataset inversion and full speed-map pipeline launch.
- A100 smoke evidence should show the full pipeline can produce `reconstruction.png`, `reconstruction_best.png`, `reconstruction_final.png`, `ground_truth.png`, `error.png`, `loss_curve.png`, `gradient_step001.png`, `gradient_step020.png`, `metadata.json`, and `run.log`.

## References and Related Code

- `src/usctbench/algorithms/fwi/kwave_adapter.py`
- `configs/algorithms/fwi_kwave_adapter.yaml`
- `configs/algorithms/fwi_kwave_full_pipeline.yaml`
- `scripts/run_fwi_kwave_full_pipeline_smoke.sh`
- `scripts/render_kwave_fwi_smoke_outputs.py`
- `tests/test_fwi_kwave_adapter.py`
- A100 reference project: `$HOME/USCT_kwave/openbreastus_diffusion/kwave_dps`
- Upstream reference: `rehmanali1994/WaveformInversionUST`, frequency-domain waveform inversion UST using a ring-array transducer.
