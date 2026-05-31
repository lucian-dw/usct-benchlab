# External Sources And Licenses

External USCT code must remain outside Git unless deliberately vendored after license review.

## Policy

- Keep full external repositories under workspace-level `external/`, which is ignored by Git.
- Prefer adapters and documented install steps over vendoring.
- Save MATLAB input `.mat` files, logs, and outputs under the run directory, not under source control.
- If an external dependency is unavailable, return `skipped` with a clear `failure_reason`.

## Optional Sources To Inspect

| Purpose | Suggested source | Git policy |
|---|---|---|
| Straight-ray SART reference | `ust-sart` | external clone only |
| Refraction-corrected GN | `rehmanali1994/refractionCorrectedUSCT.github.io` | external clone only |
| Ray-Born / r-Wave | `Ash1362/ray-based-quantitative-ultrasound-tomography` | external clone only |
| Waveform inversion references | `rehmanali1994/WaveformInversionUST` and A100 `$HOME/USCT_kwave` derivative scripts | external clone only |
| Numerical breast phantoms | NBPslices2D archive from Illinois Data Bank / A100 `$HOME/USCT_kwave` mirror | data outside Git only |

The refraction-corrected GN reference paper describes slowness reconstruction
as nonlinear least squares over travel times, solved by Gauss-Newton and
conjugate gradients, with Laplacian regularization, Bayesian formulation, and
resolution-filling gradients as stabilization options. The native
`bent_ray_gn` backend currently implements the project-standard travel-time
surrogate with Laplacian regularization and smoothing; the external MATLAB
repository remains the reference for a fuller refraction-corrected path.

The r-Wave reference data package points to the GitHub project branch/folder
`r-Wave #V1.1` and describes k-Wave-simulated transmission data for validating
ray approximations to Green's functions. The native `rwave_adapter` backend is
therefore treated as a benchmark-compatible ray-Born surrogate, not as a claim
that the full external MATLAB workflow has been vendored.

## Current Repository State

The current repository contains native Python baselines plus optional adapter paths. It does not vendor external MATLAB packages or full third-party repositories.

`bent_ray_gn` now has a project-native regularized travel-time GN smoke backend for the standard benchmark contract. The optional MATLAB path exports a standard MATLAB-readable HDF5 `.mat` input artifact under the case run directory when a MATLAB entrypoint is configured and available. The public refraction-corrected MATLAB repository remains a reference and optional external integration target; clone it outside Git before experimenting with its full MATLAB entrypoints and output ingest.

`rwave_adapter` now has a project-native regularized ray-Born/r-Wave smoke backend for the standard benchmark contract. The optional MATLAB path exports the same standard adapter input artifact for external r-Wave/ray-Born experiments. The public ray-based quantitative ultrasound tomography repository remains a reference and optional external integration target; clone it outside Git before experimenting with its full MATLAB entrypoints and output ingest.

`fwi_kwave_adapter` reads existing MATLAB v7.3 FWI result files and can optionally launch the external A100 `USCT_kwave` pipeline. The external approach is derived from `rehmanali1994/WaveformInversionUST`, which is MIT licensed upstream. Keep the full MATLAB/k-Wave code and generated FWI datasets/results outside this Git repository.

`NBPslices2D` is a CC BY numerical phantom dataset. The repository only stores
conversion code, benchmark configs, and tests; the ZIP archive, extracted `.mat`
files, converted HDF5 cases, and smoke benchmark outputs must remain under
workspace `data/` or `runs/`, both ignored by Git.

The v0.1 readiness audit executes the missing-dependency path for the MATLAB
adapter shells. A valid skip record must include `status=skipped`, a clear
`failure_reason`, `adapter_status=skipped`, and
`adapter_dependency_available=false`. CLI runs must write a standard
`failure_report.md` with `Error type: external-dependency`.

## License Checklist Before Vendoring

1. Record project URL and commit hash.
2. Record license file and compatibility assessment.
3. Confirm whether redistribution is allowed.
4. Keep large data, examples, and generated artifacts out of Git.
5. Add an entry to this document and an algorithm card update.
