# Archived Benchmark Suites

These benchmark suites are retained for development history and diagnostic
reproduction. They are not part of the v0.1 release ranking.

Archived suites include:

- `kwave_unified_quality.yaml`
- `kwave_unified_smoke.yaml`
- `kwave_real_forward_ray_smoke.yaml`
- `external_matlab_adapter_quality.yaml`
- `external_matlab_adapter_smoke.yaml`

The k-Wave unified ray/rWave path exposed observable mismatch between
wavefield-derived features and ray/surrogate inversion assumptions. v0.1 keeps
traditional methods on `travel_time_surrogate_main.yaml` and reserves k-Wave
full-wave data for `kwave_fwi_main.yaml`.

The external MATLAB adapter suites are optional integration diagnostics. They
require third-party code installed outside this repository and are not required
for the main release checks.
