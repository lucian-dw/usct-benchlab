# Observable Mismatch

## Summary

The unified k-Wave ray-feature path was useful for debugging, but it is not a
v0.1 ranking track. k-Wave-derived apparent ToF, eikonal-style features, and
complex-field prototypes did not provide a stable like-for-like input for the
travel-time surrogate solvers.

## Observed Symptoms

- Ray reconstructions could fit the provided feature residual while producing
  poor image metrics.
- Feature extraction was sensitive to arrival picking, phase convention,
  direct-arrival windows, and boundary reflections.
- True-bent and rWave complex prototypes required more validation of geometry,
  source wavelets, Green's functions, and adjoint/derivative checks before they
  could be ranked.

## v0.1 Policy

- CGLS/SIRT/SART, bent-ray surrogate, and rWave surrogate remain on
  `speedmap_travel_time_surrogate` cases.
- FWI consumes raw/precomputed k-Wave data through `kwave_fwi_main`.
- Archived k-Wave unified suites remain available under
  `configs/benchmarks/archive/` for diagnostics only.
- Experimental algorithm configs remain under `configs/algorithms/experimental/`.

Future work may reopen a dedicated complex-wavefield rWave track or a validated
true-bent ray track, but those would need independent acceptance tests and must
not be mixed into the v0.1 main benchmark table.
