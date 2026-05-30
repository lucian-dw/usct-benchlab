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
| Refraction-corrected GN | public MATLAB refraction-corrected USCT code | external clone only |
| Ray-Born / r-Wave | public ray-based quantitative ultrasound tomography code | external clone only |
| Waveform inversion references | `rehmanali1994/WaveformInversionUST` and A100 `$HOME/USCT_kwave` derivative scripts | external clone only |
| Numerical breast phantoms | NBPslices2D archive from Illinois Data Bank / A100 `$HOME/USCT_kwave` mirror | data outside Git only |

## Current Repository State

The current repository contains native Python baselines plus optional adapter shells. It does not vendor external MATLAB packages or full third-party repositories.

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
