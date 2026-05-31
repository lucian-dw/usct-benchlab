# r-Wave / Ray-Born Adapter

## Physical Assumption

This adapter targets weak-scattering ray-Born or r-Wave style reconstruction. In v0.1 the default backend is a project-native regularized ray-Born surrogate over the benchmark travel-time features. It is a bridge between ray methods and full-wave inversion, not a generative model. The optional MATLAB path remains available for external r-Wave/ray-Born code, but third-party code is not vendored.

## Input Requirements

- Standard `USCTCase` with `measurement.delta_tof_s`
- Ring geometry and grid metadata compatible with the straight-ray projector
- Optional MATLAB path: MATLAB available through `MATLAB_BIN` or `parameters.matlab_bin`, external r-Wave/ray-Born source installed outside Git, and configured `parameters.external_root` plus `parameters.entrypoint`
- When the optional MATLAB path is configured and available, the adapter writes
  a MATLAB-readable HDF5 `.mat` input file containing `USCTCase` grid,
  geometry, measurement, ground-truth, mask, and metadata fields under the case
  run directory. The configured entrypoint receives `usctbench_input_mat`,
  `usctbench_output_mat`, and `usctbench_output_dir` MATLAB variables and must
  write a standard adapter output MAT file to be ingested.

## Default Settings

- `backend: python`
- Outer ray-Born-style residual iterations: `3`
- Inner CGLS iterations: `20`
- Regularization: Laplacian, `regularization_lambda=1.0e-5`
- Smoothing: `smooth_sigma=0.35`
- Bounds: `[1300, 1700] m/s`
- Set `backend: matlab` or provide MATLAB-specific parameters to run the external dependency-checking path.

## Expected Failure Modes

- MATLAB unavailable when the MATLAB backend is requested.
- External code missing or incompatible with the local case geometry.
- Phase convention or reference-field alignment mismatch.
- Low-SNR receivers corrupting weak-scattering assumptions.
- Native smoke backend inherits the straight-ray feature approximation and does not model full complex RF scattering.

## What To Adjust First

1. Verify the travel-time sign and background speed.
2. Increase `regularization_lambda` if high-frequency ring artifacts dominate.
3. Reduce `smooth_sigma` if small inclusions disappear.
4. Validate the external package license before vendoring anything.

## Acceptance Tests

- Adapter is registered by `usct list-algorithms`.
- Native backend returns `success` with standard `sound_speed_mps`, image metrics, residual metrics, coverage metrics, and preview artifacts.
- Missing MATLAB or missing entrypoint returns `skipped`, not a crash, when the MATLAB backend is explicitly requested.
- Configured MATLAB entrypoints get `adapter_input_mat`, `adapter_output_mat`,
  `matlab_log`, and `external_entrypoint` artifacts.
- Standard adapter output MAT files are ingested into `ReconstructionResult`
  with `external_adapter_output_loaded=true`.
- CLI writes the standard failure report for skipped adapter runs.

## References and Related Code

- Adapter: `src/usctbench/algorithms/adapters/rwave.py`
- MATLAB wrapper utilities: `src/usctbench/adapters/matlab.py`
- Tests: `tests/test_matlab_adapters.py`
- External source policy and candidate code: `docs/EXTERNAL_SOURCES_AND_LICENSES.md`
- Public reference: `https://github.com/Ash1362/ray-based-quantitative-ultrasound-tomography`
