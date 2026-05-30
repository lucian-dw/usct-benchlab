# r-Wave / Ray-Born Adapter

## Physical Assumption

This optional adapter targets weak-scattering ray-Born or Rytov-style reconstruction through an external MATLAB implementation. It is a bridge between ray methods and full-wave inversion, not a generative model.

## Input Requirements

- Standard `USCTCase`
- MATLAB available through `MATLAB_BIN` or `parameters.matlab_bin`
- External r-Wave/ray-Born source installed outside Git
- Configured `parameters.external_root` and `parameters.entrypoint`

## Default Settings

The default config is intentionally non-executing. It returns `skipped` until MATLAB and the external entrypoint are explicitly configured.

## Expected Failure Modes

- MATLAB unavailable.
- External code missing or incompatible with the local case geometry.
- Phase convention or reference-field alignment mismatch.
- Low-SNR receivers corrupting weak-scattering assumptions.

## What To Adjust First

1. Use lower frequency and a better background model.
2. Reject low-SNR receivers.
3. Check complex phase convention and reference alignment.
4. Validate the external package license before vendoring anything.

## Acceptance Tests

- Adapter is registered by `usct list-algorithms`.
- Missing MATLAB or missing entrypoint returns `skipped`, not a crash.
- CLI writes the standard failure report for skipped adapter runs.

