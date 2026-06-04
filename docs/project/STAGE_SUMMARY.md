# v0.1 Stage Summary

## Current Release Boundary

`usct-benchlab` v0.1 is a benchmark harness release, not a new algorithm
release. The repository now separates two main tracks:

- `travel_time_surrogate_main`: classical CGLS/SIRT/SART, bent-ray surrogate,
  and rWave surrogate on property-map-derived travel-time features.
- `kwave_fwi_main`: k-Wave full-wave data and external pure FWI through
  `fwi_kwave_adapter`.

## Frozen Decisions

- Do not continue tuning k-Wave-derived ray/rWave/true-bent image quality for
  v0.1.
- Do not add diffusion or other generative reconstruction.
- Keep k-Wave full-wave data for FWI.
- Keep traditional algorithms as fast, classical baselines on travel-time
  surrogate data.
- Preserve k-Wave ray/rWave experiments as diagnostics only.

## Release Evidence

The release checks should verify:

- package import and CLI registration;
- unit tests for schema, metrics, projectors, algorithms, adapters, and docs;
- no tracked raw data or run outputs;
- README comparison panels for OpenBreastUS and NBPslice2D;
- clear benchmark tier and provenance documentation.

Heavy A100 benchmark reruns are outside the local cleanup commit unless the
user explicitly requests fresh runtime evidence.
