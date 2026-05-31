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
- `configs/algorithms/fwi_kwave_full_pipeline.yaml` is the A100 smoke config for `full128`, RF travel-time warm start, 0.3 MHz, 20 sound-speed iterations, no attenuation inversion, velocity clamp `[1350, 1600]` m/s, and `save_raw_grad_iters: 20`.
- Output images are resized to the input case grid for benchmark metrics and artifact writing.

## Expected Failure Modes

- Missing external result file.
- MATLAB result missing `VEL_ESTIM`.
- Result/case mismatch if `result_path` belongs to a different k-Wave dataset than the input case.
- External launch failure due to missing MATLAB, missing CUDA/MEX binaries, missing k-Wave paths, or absent generated datasets.
- Non-decreasing FWI loss on unstable schedules.

## What To Adjust First

- Verify `result_path`, `dataset_path`, `mat_path`, `mat_key`, and `sample_index` refer to the same case.
- Start with an existing validated result before enabling `run_external`, then move to `invert_existing_dataset`, then `full_pipeline_from_speed_map`.
- Use low starting frequency, small iteration counts, and conservative update damping.
- Use warm-starts from the external project when available.
- If image RMSE worsens while waveform loss decreases, reduce iterations, clamp per-step updates, or use a better multi-frequency schedule before claiming reconstruction quality.
- Check the external stdout/stderr log before changing inversion parameters.

## Acceptance Tests

- Unit test reads a synthetic MATLAB v7.3 FWI result and returns a successful `ReconstructionResult`.
- Missing result path skips clearly rather than crashing.
- Unit tests verify command construction for both existing-dataset inversion and full speed-map pipeline launch.
- A100 smoke evidence should show the full pipeline can produce `reconstruction.png`, `ground_truth.png`, `error.png`, `loss_curve.png`, `gradient_step001.png`, `gradient_step020.png`, `metadata.json`, and `run.log`.

## References and Related Code

- `src/usctbench/algorithms/fwi/kwave_adapter.py`
- `configs/algorithms/fwi_kwave_adapter.yaml`
- `configs/algorithms/fwi_kwave_full_pipeline.yaml`
- `scripts/run_fwi_kwave_full_pipeline_smoke.sh`
- `scripts/render_kwave_fwi_smoke_outputs.py`
- `tests/test_fwi_kwave_adapter.py`
- A100 reference project: `$HOME/USCT_kwave/openbreastus_diffusion/kwave_dps`
- Upstream reference: `rehmanali1994/WaveformInversionUST`, frequency-domain waveform inversion UST using a ring-array transducer.
