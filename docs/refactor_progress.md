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

## P2: joint WUST protocol migration

- Adapter sends one v2 reconstruction request with frequency input, initial map,
  physical Hz schedule and effective update cap. WUST alone maps sorted
  frequencies and sampling indices. Static discovery no longer launches a probe.
- Removed source/build/dirty admission, input snapshots, array/file hashes and
  whole-artifact write/readback. Data splits keep actual masks, indices and seeds.
- WUST field definitions generate the seven schemas used by its serializers;
  canonical artifacts no longer repeat derived grid coordinates. Optional model
  history is disabled by default. Existing slowness updates are unchanged.
- Shared non-square fixture powers the one cross-repository validation script.
  GT independence stays in the light adapter test, not duplicate MATLAB runs.
- Local full pytest: 710 passed, 1 skipped (25.06 s). Ruff passed; Black check
  passed for 126 source/test/integration files; git diff --check passed.
- A100 v2 CPU and GPU gates passed on 23x27, two TX/four RX, two updates from a
  three-entry plan. Each used one fixture-generation and one reconstruction
  MATLAB invocation. CPU reconstruction 11.48 s; GPU 15.30 s including startup.
  Artifacts: /tmp/usct_refactor_v2_cpu and /tmp/usct_refactor_v2_gpu on A100.
- A100 CPU/GPU FD, batching, resampling, ingestion, budget/failure tests passed;
  log /tmp/usct_refactor_v2_math.log. WUST Python suite: 15 tests passed.

Next: P3 archives the explicitly listed historical tools/assets outside both
repositories and removes repeated CI/compatibility entrypoints. No new imaging
campaign was run; these are small interface and numerical regression fixtures.

## P3: source/research separation

- Moved 85 explicitly listed research files into the sibling archive; originals
  of deleted/merged files were also preserved there. `docs/research_index.md`
  records locations and source commits. Original data, runs and user checkout
  were not touched.
- Removed auto-edit CI, duplicate audit/smoke workflows, Tiny FWI and legacy
  operator import wrappers. Default CI runs Black, Ruff and one CPU pytest.
- Kept independent adjoint, Taylor/FD, Snell and dense Born remainder checks in
  their owning operator/solver tests. Removed source-inspection and historical
  report certification tests. Simulation, rendering and quickstart entrypoints
  now have separate responsibilities.
- Local and A100 full pytest: 584 passed (20.08 s / 9.12 s). WUST Python: 15
  passed. Ruff, Black and diff checks passed. Synthetic CGLS quickstart wrote
  result.h5, metrics.json, metadata.yaml and preview.png successfully.
- Running the old broad synthetic suite with the formal Bent config exposed
  an existing roundoff-scale homogeneous-case line-search failure (initial
  residual 7.1e-20). A trial classification change was rejected because it also
  hid deliberately unusable inner directions; no mathematical assertions were
  removed. Resolve this at P5, not by declaring absolute-gradient convergence.

Next: P4 unifies typed configuration, host limits, static discovery and outputs.

## P4: persisted data, provenance and installation

Completed: measurement HDF5 reads/writes now share an explicit axis/unit table;
pressure units remain acquisition facts and unknown units are not invented.
Removed unused result placeholders and inferred provenance/risk flags. Pillow
owns optional preview encoding. Python dependencies now have one source in
pyproject.toml, with installation callers migrated from requirements.txt.

Validation: A100 full CPU suite passed (586 tests, 9.34 s); the added rectangular
complex-data roundtrip covers masks, valid zeros, units and rejected axis labels.
The provenance regression confirms that free text does not imply acquisition
facts. Black, Ruff and diff whitespace checks passed for this unit.

Next: replace legacy config aliases and dict roundtrips with one prepared typed
configuration, independent deployment settings and resolved run controls.
