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
| Waveform inversion references | classical waveform inversion USCT code | external clone only |

## Current Repository State

The current repository contains only native Python baselines and optional adapter shells. It does not vendor external MATLAB packages or full third-party repositories.

## License Checklist Before Vendoring

1. Record project URL and commit hash.
2. Record license file and compatibility assessment.
3. Confirm whether redistribution is allowed.
4. Keep large data, examples, and generated artifacts out of Git.
5. Add an entry to this document and an algorithm card update.

