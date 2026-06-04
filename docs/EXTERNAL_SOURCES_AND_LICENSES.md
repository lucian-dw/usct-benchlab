# External Sources and Licenses

External datasets, MATLAB packages, k-Wave environments, and generated
benchmark outputs should stay outside this Git repository.

## Repository Policy

- Keep full external repositories under a workspace-level `external/` directory.
- Keep raw datasets under `data/`.
- Keep generated outputs under `runs/`.
- Keep checkpoints or solver weights under `checkpoints/`.
- Do not commit `.mat`, `.h5`, `.npy`, `.npz`, `.pt`, `.ckpt`, or raw run
  artifacts.
- Prefer small adapters and documented setup steps over vendoring third-party
  code.

## Optional External Sources

| Purpose | Suggested source | Git policy |
| --- | --- | --- |
| Straight-ray references | USCT SART and algebraic tomography literature | cite only |
| Refraction-corrected tomography | Refraction-corrected USCT papers and code | external checkout only |
| Ray-Born / rWave references | Ray-based quantitative ultrasound tomography work | external checkout only |
| Full waveform inversion | k-Wave and WaveformInversionUST-style workflows | external environment only |
| OpenBreastUS | OpenBreastUS dataset distribution | data outside Git |
| NBPslice2D | 2D Acoustic Numerical Breast Phantoms for USCT | data outside Git |

## v0.1 Scope

`usct-benchlab` v0.1 contains native Python baselines, dataset conversion
helpers, a common benchmark harness, and an FWI result adapter. It does not
vendor external MATLAB packages or full third-party repositories.

`bent_ray_gn` is a regularized bent-ray-style travel-time baseline in v0.1. The
full external eikonal/refraction-corrected solver remains outside this package.

`rwave_adapter` is an rWave/ray-Born-inspired adapter baseline in v0.1. A full
complex ray-Born reproduction requires external complex frequency-domain data
and solver code.

`fwi_kwave_adapter` can ingest external k-Wave/FWI result artifacts and report
them using the package-standard `result.h5`, `metrics.json`, `metadata.yaml`,
and `preview.png` outputs.

## License Checklist Before Vendoring

1. Record the project URL and commit hash.
2. Record the license file and compatibility assessment.
3. Confirm redistribution is allowed.
4. Keep large data, examples, and generated artifacts out of Git.
5. Add a focused note to this document and update the relevant algorithm docs.
