# rWave / Ray-Born Adapter

The current v0.1 `rwave_adapter` is a travel-time surrogate baseline. It uses
the same `USCTCase` travel-time feature contract as the straight-ray and
regularized bent-ray surrogate algorithms.

## Current Mainline

- Suite: `configs/benchmarks/travel_time_surrogate_main.yaml`
- Algorithm config: `configs/algorithms/rwave_adapter.yaml`
- Measurement provenance: `speedmap_travel_time_surrogate`
- `uses_kwave_wavefield=false`
- `uses_complex_wavefield=false`
- `surrogate_travel_time_backend=true`

This is not a validated full rWave or Ray-Born implementation. It is kept as a
traditional-method comparison point under the travel-time surrogate track.

## Retired k-Wave Complex Path

The previous native Python complex-wavefield prototype and full-Green MATLAB
experiments are not part of the current mainline. k-Wave raw/precomputed data
is reserved for `configs/benchmarks/kwave_fwi_main.yaml` and
`fwi_kwave_adapter`.

If a future rWave track is reopened, it should be implemented as a separate
benchmark with an explicit complex pressure/reference contract, source
wavelet convention, Green's function validation, derivative/adjoint checks,
and independent acceptance metrics. It should not be mixed into the FWI
mainline or presented as a travel-time surrogate result.
