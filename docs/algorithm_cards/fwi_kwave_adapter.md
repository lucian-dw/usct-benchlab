# k-Wave FWI Adapter

## Physical Assumption

This adapter represents frequency-domain waveform inversion for ring-array USCT using an external k-Wave/WaveformInversionUST-style Helmholtz workflow. It treats the external MATLAB result as the authoritative full-wave inversion output and maps it back into the `USCTCase -> ReconstructionResult` contract.

## Input Requirements

- A standard `USCTCase` with sound-speed ground truth when image metrics are required.
- A MATLAB v7.3 result file containing `VEL_ESTIM`; optional datasets include `ATTEN_ESTIM`, `LOSS_ITER`, `psnr_value`, `ssim_value`, and `datasetPath`.
- For active external execution, an A100 environment with the `USCT_kwave` project, k-Wave, MATLAB engine or MATLAB batch support, and the required generated k-Wave dataset.

## Default Settings

- Default mode is result ingestion only: `run_external: false`.
- `result_path` must point to an existing k-Wave FWI result `.mat`.
- `run_external: true` calls `openbreastus_diffusion.kwave_dps.run_full_pipeline` with explicit pipeline arguments from the config.
- Output images are resized to the input case grid for benchmark metrics and artifact writing.

## Expected Failure Modes

- Missing external result file.
- MATLAB result missing `VEL_ESTIM`.
- Result/case mismatch if `result_path` belongs to a different k-Wave dataset than the input case.
- External launch failure due to missing MATLAB, missing CUDA/MEX binaries, missing k-Wave paths, or absent generated datasets.
- Non-decreasing FWI loss on unstable schedules.

## What To Adjust First

- Verify `result_path` and `dataset_path` refer to the same case.
- Start with an existing validated result before enabling `run_external`.
- Use low starting frequency, small iteration counts, and conservative update damping.
- Use warm-starts from the external project when available.
- Check the external stdout/stderr log before changing inversion parameters.

## Acceptance Tests

- Unit test reads a synthetic MATLAB v7.3 FWI result and returns a successful `ReconstructionResult`.
- Missing result path skips clearly rather than crashing.
- A100 smoke evidence should show the adapter can ingest an existing k-Wave FWI result and write standard benchmark artifacts.

## References and Related Code

- `src/usctbench/algorithms/fwi/kwave_adapter.py`
- `configs/algorithms/fwi_kwave_adapter.yaml`
- `tests/test_fwi_kwave_adapter.py`
- A100 reference project: `$HOME/USCT_kwave/openbreastus_diffusion/kwave_dps`
- Upstream reference: `rehmanali1994/WaveformInversionUST`, frequency-domain waveform inversion UST using a ring-array transducer.
