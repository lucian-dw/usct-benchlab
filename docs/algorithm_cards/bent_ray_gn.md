# Bent-Ray Gauss-Newton Adapter

## Physical Assumption

This adapter targets refraction-corrected travel-time tomography. In v0.1 the default backend is a project-native regularized Gauss-Newton travel-time surrogate that uses the standard `USCTCase` travel-time features, CGLS inner solves, Laplacian regularization, and smoothing. It preserves an optional MATLAB path for the public refraction-corrected USCT code, but the smoke/quality benchmark path does not vendor or require external code.

## Input Requirements

- Standard `USCTCase` with `measurement.delta_tof_s`
- Ring geometry and grid metadata compatible with the straight-ray projector
- Optional MATLAB path: MATLAB available through `MATLAB_BIN` or `parameters.matlab_bin`, external source installed outside Git, and configured `parameters.external_root` plus `parameters.entrypoint`
- When the optional MATLAB path is configured and available, the adapter writes
  a MATLAB-readable HDF5 `.mat` input file containing `USCTCase` grid,
  geometry, measurement, ground-truth, mask, and metadata fields under the case
  run directory. The configured entrypoint receives `usctbench_input_mat`,
  `usctbench_output_mat`, and `usctbench_output_dir` MATLAB variables and must
  write a standard adapter output MAT file to be ingested.
- The case run directory also receives first-party MATLAB helper functions:
  `usctbench_read_case.m` and `usctbench_write_result.m`. A refraction-corrected
  entrypoint can call these helpers, then use the public repository's
  `ctReconLaplacianRegularized.m` pattern and `Functions/eikProjMat.m` /
  `Functions/laplacianOperator.m` internals without re-implementing the
  usct-benchlab HDF5 contract.

## Default Settings

- `backend: python`
- Outer Gauss-Newton-style iterations: `4`
- Inner CGLS iterations: `16`
- Regularization: Laplacian, `regularization_lambda=3.0e-5`
- Smoothing: `smooth_sigma=0.6`
- Bounds: `[1300, 1700] m/s`
- Set `backend: matlab` or provide MATLAB-specific parameters to run the external dependency-checking path.

## Expected Failure Modes

- MATLAB unavailable when the MATLAB backend is requested.
- External source code is missing or not licensed for vendoring.
- The external package expects a different geometry or input `.mat` layout.
- Refraction ray tracing diverges from a poor initial model.
- Native smoke backend over-smooths small high-contrast inclusions.

## What To Adjust First

1. Verify geometry and travel-time sign convention.
2. Increase `regularization_lambda` or `smooth_sigma` if ring artifacts dominate.
3. Increase `inner_iterations` before increasing outer iterations.
4. Configure the external package path and entrypoint only after the native smoke backend passes.

## Acceptance Tests

- Adapter is registered by `usct list-algorithms`.
- Native backend returns `success` with standard `sound_speed_mps`, image metrics, residual metrics, coverage metrics, and preview artifacts.
- Missing MATLAB or missing entrypoint returns `skipped`, not a crash, when the MATLAB backend is explicitly requested.
- Configured MATLAB entrypoints get `adapter_input_mat`, `adapter_output_mat`,
  `adapter_contract_dir`, `matlab_log`, and `external_entrypoint` artifacts.
- Standard adapter output MAT files are ingested into `ReconstructionResult`
  with `external_adapter_output_loaded=true`.
- If a MATLAB entrypoint writes only `sound_speed_mps`, Python augments the
  result with project-standard image, water-baseline, and data-residual metrics
  before benchmark assessment.
- A failure report is written by the CLI for skipped runs.

## References and Related Code

- Adapter: `src/usctbench/algorithms/adapters/refraction_gn.py`
- MATLAB wrapper utilities: `src/usctbench/adapters/matlab.py`
- Tests: `tests/test_matlab_adapters.py`
- External source policy and candidate code: `docs/EXTERNAL_SOURCES_AND_LICENSES.md`
- Public reference: `https://github.com/rehmanali1994/refractionCorrectedUSCT.github.io`
