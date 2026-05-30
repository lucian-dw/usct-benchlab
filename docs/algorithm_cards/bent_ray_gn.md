# Bent-Ray Gauss-Newton Adapter

## Physical Assumption

This optional adapter targets refraction-corrected travel-time tomography. It is intended to wrap established MATLAB code rather than reimplement a full bent-ray solver in v0.1.

## Input Requirements

- Standard `USCTCase`
- MATLAB available through `MATLAB_BIN` or `parameters.matlab_bin`
- External MATLAB source installed outside Git, usually under `external/`
- Configured `parameters.external_root` and `parameters.entrypoint`

## Default Settings

The default config does not execute external code. It returns `skipped` with a clear reason until MATLAB and an entrypoint are configured.

## Expected Failure Modes

- MATLAB is unavailable on the active machine.
- External source code is missing or not licensed for vendoring.
- The external package expects a different geometry or input `.mat` layout.
- Refraction ray tracing diverges from a poor initial model.

## What To Adjust First

1. Start from a smoothed straight-ray reconstruction.
2. Verify geometry and units before tuning regularization.
3. Configure the external package path and entrypoint.
4. Save MATLAB stdout/stderr logs under the run directory when enabling execution.

## Acceptance Tests

- Adapter is registered by `usct list-algorithms`.
- Missing MATLAB or missing entrypoint returns `skipped`, not a crash.
- A failure report is written by the CLI for skipped runs.

## References and Related Code

- Adapter: `src/usctbench/algorithms/adapters/refraction_gn.py`
- MATLAB wrapper utilities: `src/usctbench/adapters/matlab.py`
- Tests: `tests/test_matlab_adapters.py`
- External source policy and candidate code: `docs/EXTERNAL_SOURCES_AND_LICENSES.md`
