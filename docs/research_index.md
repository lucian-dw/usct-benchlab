# Research archive

Maintained code implements reconstruction, data import and evaluation. Historical
experiments are preserved separately, not run by default CI or treated as current
algorithm defaults.

## Location and versions

The archive is a sibling of the two repositories:
`research_archive/usct-benchlab/`. Every moved file retains its original relative
path. It was extracted from BenchLab `22cd54c` and the P1/P2 working tree ending
at `dd7e467`. The original Git history also retains these materials.

This external archive is not shipped by `pip install` or a normal Git clone.
To replay old experiments, use their original Git revision and recorded Python,
MATLAB/k-Wave and A100 environment. Current APIs are not promised to run historical
scripts unchanged. Original raw data, run directories and generated results were
not moved or deleted. Their locations remain in the archived experiment records.

| Topic | Preserved material | Interpretation |
|---|---|---|
| ToF attribution/adaptation | `scripts/audit_tof_*`, `tof_audit_common.py`, `docs/validation/2026-09-08_*`, `2026-09-09_*` | Includes matched surrogate, k-Wave-derived and diagnostic controls; consult each record's actual input rather than a blanket GT-generated label. |
| Finite-frequency delays | `scripts/run_finite_frequency_traveltime.py`, `_finite_frequency_trf.py`, phase/arrival modules and their tests | Research peak/phase/TRF objectives, not registered production reconstruction. |
| Independent solver research | `_inverse_solver_audit.py`, `audit_tof_coarse_optimum.py`, augmented audit figures/tables | Small dense, adjoint and directional-derivative references remain in maintained tests; full parameter scans are archived. |
| Reconstruction panels | `docs/assets/reconstruction/`, `docs/readme_results.md`, `scripts/render_readme_results.py` | Historical OpenBreastUS HET/FIB/FAT/EXD and NBPslice2D A/B/C/D results. Matched and k-Wave panels have different observation assumptions; historical FWI may use separate acquisition and coordinates. They are not a single current ranking or a default-parameter recipe. |
| Earlier figures and stage reports | `docs/assets/`, `docs/review/`, `docs/validation/` | Retained positive and negative results, stop reasons, captions, original configurations and run locations. |
| Teaching/old infrastructure | copied Tiny FWI and removed workflow/script sources | Historical reference only; no production entry point. |

Browse the [local archive](../../research_archive/usct-benchlab/).
Current examples are in [examples/README.md](../examples/README.md).
