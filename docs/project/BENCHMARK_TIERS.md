# Benchmark Tiers

v0.1 uses two main benchmark tracks.

## Track A: Travel-Time Surrogate Main

`configs/benchmarks/travel_time_surrogate_main.yaml` is the canonical
traditional-method track. It runs:

- `straight_cgls`
- `straight_sirt`
- `straight_sart`
- `bent_ray_gn`
- `rwave_adapter`

The measurement provenance is `speedmap_travel_time_surrogate`. These cases
are generated from property maps or oracle travel-time features and are useful
for solver sanity, stability comparisons, and fast regression testing. They
must not be reported as full wavefield inversion evidence.

Expected metadata:

- `benchmark_type=travel_time_surrogate`
- `measurement_provenance=speedmap_travel_time_surrogate`
- `uses_kwave_wavefield=false`
- `uses_complex_wavefield=false`
- `inverse_crime_risk=high`

## Track B: k-Wave FWI Main

`configs/benchmarks/kwave_fwi_main.yaml` is the canonical k-Wave track. It
runs only:

- `fwi_kwave_adapter`

The FWI adapter consumes raw/precomputed k-Wave channel data or external
preprocessed k-Wave MAT files. The current mainline is bulk-support pure FWI
aligned to the A100 `openbreastus_diffusion/kwave_dps` implementation, without
diffusion or generative priors.

Expected metadata:

- `benchmark_type=kwave_fwi_main`
- `measurement_provenance=self_simulated_kwave_wavefield_or_precomputed_kwave_mat`
- `uses_kwave_wavefield=true`
- `uses_complex_wavefield=true`
- `inverse_crime_risk=medium`

## Retired From Mainline

The previous unified k-Wave ray-feature path is retired from the main
benchmark. In particular, do not run straight-ray, bent-ray surrogate,
attenuation, or rWave surrogate algorithms on k-Wave-derived apparent/eikonal
ToF feature cases as a formal result. The compatibility file
`configs/benchmarks/kwave_unified_quality.yaml` now points to FWI only and
exists only to avoid stale entry-point confusion.

The following paths are no longer mainline:

- k-Wave apparent/eikonal ToF feature cases for ray algorithms
- true-bent shortest-path/eikonal prototypes
- finite-frequency ToF prototypes
- rWave complex/full-Green k-Wave prototypes

They can be revisited later as separate research tracks, but the current
repository state intentionally keeps traditional methods on travel-time
surrogate data and k-Wave data on FWI.
