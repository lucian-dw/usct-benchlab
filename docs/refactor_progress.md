# P1-P6 refactor execution record

The approved plan and per-file actions are in the maintainer's
`USCT_REFACTOR_HANDOFF` development package. This document records implementation,
not completion of the entire plan. Both repositories use branch
`codex/full-refactor-p1-p6`; the original worktrees and experiment data are untouched.

## P1: data and result boundaries

Implemented:

- Continuous property maps and nearest-neighbor labels now sample the same
  cell-center coordinates, preserving physical cell-edge FOV for rectangular and
  non-integer resizing. Coordinate-based grids use cell-edge origins.
- MAT reader format selection no longer intercepts exceptions raised by consumers.
- OpenBreastUS conversion overwrites only selected outputs; unrelated HDF5 files
  are no longer removed by a directory-wide cleanup.
- Pressure imports resample GT using actual source/target coordinates and reject
  out-of-FOV grids. Time-only imports validate sampling without a dummy DTFT.
- Speed-volume axis order, sample axis and units are declared preparation inputs.
  Missing k-Wave coordinates fail instead of inventing spacing; the reference
  speed is explicit, not computed from GT. Empty NBP labels are marked missing.
- Born preparation broadcasts scalar/frequency source calibration to frequency/TX.
- `max_backtracks=N` now means one initial trial plus at most N halvings. The
  zero-backtrack path is exercised against the independent dense reference.
- The runner gives algorithms a shallow case with empty GT, saves numerical
  arrays before evaluation/preview, and records separate stage outcomes. Reporting
  failures no longer replace successful reconstructed arrays. Wall time is measured
  even on failures; `process_peak_rss_mib` is the process-lifetime RSS high-water mark.
- Gradient stationarity has its own termination category. Local SSIM without a
  complete window is unavailable rather than replaced with global SSIM. Perfect
  reconstruction PSNR is JSON null with `psnr_status=perfect_reconstruction`.
- Dead residual/baseline helpers and the image-mask digest were removed; current
  callers were updated. Historical report scripts remain until P3 archival.

Validation so far:

- Latest local full pytest: 719 passed, 1 skipped (25.19 seconds).
- Ruff passed; Black applied; `git diff --check` passed.
- Initial A100 full run: 714 passed, 3 failed because the transferred tree had no
  Git index and the host used ASCII text decoding. Acceptance is being rerun with
  a local source index and `PYTHONUTF8=1`: latest run 720 passed (29.55 seconds).

Next: P2 jointly migrates WUST and this adapter to protocol v2. Legacy automatic
case selection is explicitly left to P4; reader decomposition and shared
surrogate assembly belong to P5. No P2-P6 completion is claimed at this checkpoint.

## Cross-repository order

P2 must migrate both sides together; do not remove BenchLab runtime identity
checks independently while WUST still emits protocol v1. P3 then moves research
assets to sibling `research_archive/`; P4 unifies configuration/controls; P5 keeps
the specified existing numerical update ordering; P6 consolidates docs and tests.
