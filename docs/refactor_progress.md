# P1-P6 refactor execution record

The approved plan and per-file actions are in the maintainer's
`USCT_REFACTOR_HANDOFF` development package, now copied to the sibling
research_archive/handoff directory. P1-P6 implementation and the small acceptance
gates are complete; earlier checkpoints below retain their original scope.
Both repositories use branch
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

## P4: static algorithm discovery

Completed: importing usctbench now exposes the maintained algorithms from one
static factory/specification table. Removed repeated CLI registration, mutable
replacement/clear hooks and the untyped legacy-description fallback. Algorithms
no longer register themselves; callers and tests use the ordinary package API.

Validation: full local CPU suite 586 passed (19.94 s), Black and Ruff passed.
The public-import test instantiates every maintained entry without CLI setup.
No numerical implementation or defaults changed in this unit.

Next: make the configuration payload stay typed through execution and unify
request budgets with independent host limits.

## P4: typed execution configuration

Completed: `prepare_config` now supplies concrete parameter objects to every
algorithm. Request budgets and production stopping use one `RunControls`;
independent host limits intersect the resolved request, including zero budgets.
Deployment paths/backend and evaluation policies have separate typed fields.
Born variants use one selector. Removed aliases, `BudgetCaps`, `StopPolicy` and
the dict roundtrip implementation. YAML, API, schemas and tests were migrated.

Validation: local full CPU suite 564 passed (21.39 s); A100 564 passed (9.34 s).
Black, Ruff and `git diff --check` passed. Independent optimization references,
hold-out isolation, adjoints and native reconstruction assertions remain.
Historical alias acceptance tests were replaced by canonical admission tests.
Migration instructions are in `docs/parameter_contract.md`.

Next: shorten result publication/reporting and unify CLI/public execution. Then
P5 consolidates the mathematical kernels without changing their update order.

## P4: public execution and publication

Completed: one `run/suite/evaluate` API and CLI handler dispatch. Output paths
are explicit, suite paths are relative to the suite file, and GT is removed
from the numerical view. Arrays and run records precede the success index;
evaluation/preview failures keep saved arrays. Statuses are completed,
budget_exhausted or failed. Partial deployment overrides preserve other explicit
settings. Reusing an output directory invalidates only generated reporting files.

Validation: local CPU suite 564 passed (20.91 s); A100 preceding execution unit
563 passed (7.19 s), with the final deployment regression included in the next
remote gate. Black, Ruff and whitespace checks passed. CLI synthetic quickstart
and demo produced arrays, run records, metrics and previews. A100 real MATLAB
CPU protocol gate passed on a 23x27 grid: one fixture invocation (10.89 s), one
two-update reconstruction invocation (10.79 s), stop reason max_iterations.

Next: P5 removes duplicate physical kernels and consolidates numerical solvers.

## P5: mathematical kernels and preparation

Completed: one CSR straight-ray operator; one interpreted/Numba marching kernel;
explicit Ray-Born physics options and canonical (frequency,TX) source arrays;
one masked weighted water calibration; shared property-map surrogate assembly.
Public data writers now use prepare names and explicit case/slice selection.
Projected CG receives bounds/ROI directly. Linearized normal-CG/LSMR/LSQR share
one linear module and return a step plus LinearSolveInfo. Fine-grid regularizers
live with operators. Phase initialization uses training-only weights.
Numerical failures retain their original message and last complete iterate.

The only discrete numerical correction is homogeneous Eikonal cancellation:
subtract numerical object/water times before adding the geometric water time.
This removes roundoff-only line-search failure at the exactly homogeneous model.
All other changes preserve update formulas and trial projection order.

Validation: local CPU 462 passed (16.27 s); A100 CPU 462 passed (6.80 s).
The suite includes rectangular weighted complex LS, rank deficiency, independent
bounded/Huber optima, discrete adjoints/FD and full nonlinear objective checks.
Ruff and git diff --check pass; Black formatted the changed modules.

Next: finish WUST P5 hardware validation, then P6 consolidates test ownership,
current docs, examples and final installation/integration acceptance.

## P6: current docs, test ownership and final acceptance

Implemented:

- Rewrote both READMEs and current usage, data, math, parameters, Agent, operator,
  evaluation and development docs. Added one migration changelog. Historical
  pictures remain in the external archive, not default-setting claims.
- OpenBreastUS now takes explicit files and slice indices through CLI and Python;
  it does not infer four classes or create symlink/index workarounds. This closes
  the remaining P4 automatic-selection path found while verifying the new docs.
- Reports retain the observation domain and units beside provenance/model. ToF
  and pressure residuals are not presented as a shared score. Missing GT remains
  an unavailable metric, not a fabricated number.
- Consolidated tests into their owning operator, solver, I/O, evaluation and CLI
  modules. Kept independent LS/KKT/adjoint/FD/hold-out references; removed repeated
  budget matrices, one duplicate dense Born remainder and historical interface
  assertions. Archived FMM grid-refinement studies and retained one Snell case
  plus derivatives for both spatial orders.
- The manual two-repo gate accepts an existing coherent fixture, enabling CPU/GPU
  comparison with zero new fixture-generation calls. Production reconstruction
  is still a single MATLAB invocation. No new imaging campaign was launched.

Executed validation:

- Fresh Python 3.12 environment without system-site packages: editable dev/viz
  installation succeeded; pip check reports no broken requirements; full pytest
  **346 passed in 13.06 s**. Installed CLI discovery/help, synthetic quickstart,
  evaluate and the three-algorithm synthetic suite succeeded.
- A100 Python 3.13: **346 passed in 4.31 s** with the intended source tree.
  An initial invocation outside the repo could not import tests.helpers; rerunning
  with the repo as cwd corrected the invocation, not the numerical code.
- Black: 87 maintained Python files pass; Ruff, compileall and diff checks pass.
- A100 Full-Green CPU/GPU forward relative difference 1.34297e-13; Jacobian
  difference 1.26625e-11; GPU real-inner-product adjoint error 1.59902e-16.
- BenchLab -> WUST CPU gate: one reconstruction process, two updates, 23x27
  non-square grid, 11.09 s including startup. GPU: the same saved fixture,
  one reconstruction process, 15.25 s. Both report max_iterations, not convergence.
- WUST CPU/GPU gradient, source batching, RF ingestion, FIR and native independent
  simulation gates passed; details are in WUST's execution record. CPU and GPU
  gates saved /tmp/usct_refactor_p6_integration_{cpu,gpu}/gate.json on A100.
- Local evidence copies are in sibling evidence/p6. No tracked scientific arrays,
  runtime binaries or files over 2 MB. Seven generated WUST schemas are unchanged
  by regeneration. Public docs have no personal absolute paths or broken local
  links (the external archive intentionally requires the sibling directory).

Environment notes: the first system-site installation inherited an unrelated
Conda solver conflict; the clean environment above passes. Earlier rsync copies
retained removed source files; the exact removed-file allowlist was moved into
the isolated A100 research_archive/sync_cleanup, without touching original jobs,
data or checkouts. Neither environment issue was hidden by changing assertions.

Next: maintainer review of the two local refactor branches. No push, PR, merge,
large-data rerun or clinical validation is included in this delivery.

## Final taskbook F02

- Required explicit dataset_name through generic volume reader, preparation and
  CLI. Removed largest-array selection and the old k-Wave-to-GT-surrogate path.
- Pressure import keeps observations when optional C coordinates are missing;
  explicit truth_grid enables label registration. Standard RF axes are now
  consistently documented as time,tx,rx and the extra example transpose is gone.
- Focused validation: `pytest -q tests/data/test_waveforms.py
  tests/data/test_synthetic_and_conversion.py tests/cli`: 60 passed (2.97 s).
  Includes multi-array selection, asymmetric writer axes, independent labels,
  explicit coordinates, saved round-trip and CLI regressions. Ruff/Black passed
  on changed Python files.
- Next: F03 training-frequency support and F04 anisotropic Green discretization.

## Final taskbook F03-F04

- Phase unwrap now rejects channels outside the training-frequency/physical-delay
  support and records unwrap_range_rejected_channels. The 8 us -> -2 us alias
  has zero confidence, without inspecting held-out observations.
- Nonzero Green displacements use their true distance. Only the self cell uses
  the equal-area disk integral; FFT padding is explicitly zero.
- Focused arrival/phase and Born operator suite: 26 passed (0.66 s). The Green
  test compares a complex field on an anisotropic rectangular grid against an
  independently assembled dense convolution and quadrature self term, with
  2e-14 absolute/relative tolerance. Existing derivative tests remain intact.
- Next: F05 shared bounded real/complex solve and explicit IRLS step budgets.

## Final taskbook F05

- Fixed Born now uses projected_cg with a constant pressure offset, real-model
  complex-data gradient and explicit squared-slowness bounds. Deleted the old
  clipped linear_cgls implementation and its unused projection closure.
- IRLS consumes irls_max_inner_steps directly; default 30/3 migrated to 10.
  Training-only refreshes, activity restarts and true Huber acceptance remain.
- `pytest -q tests/solvers/test_projected_cg.py tests/algorithms`: 144 passed
  (9.50 s). Includes the two-pixel complex analytic solution [0, 1e-9/65],
  monotone true objective, independent Huber/box references and existing native
  algorithm regressions. An initial command named nonexistent test files and
  collected nothing; the corrected directory-based command above was executed.
- Next: F06 explicit operator data dtype and native linear-solver diagnostics.

## Final taskbook F06

- Straight/Eikonal/Born CPU/GPU Jacobians declare data_dtype; reduced operators
  propagate it. Complex Jacobian with real residual now keeps imaginary rows.
- Removed post-solve J/J* recomputation; inner diagnostics use native iterative
  estimates. The later F08 unit centralizes their scale and scope semantics.
- Focused linear/nonlinear, Bent, relinearized Born and reduced-model tests:
  86 passed (2.60 s). An initial run found nine test doubles missing the newly
  required dtype; their interfaces were migrated without removing assertions.
- Next: F07 WUST update validation, then F08 ordinary scalar normalization.

## Final taskbook F08

- Replaced exponent scaling with rhs_scale/operator_scale for the complete
  real-stacked augmented system. Normal CG, LSMR and LSQR share preparation,
  optional once-per-subproblem column_rms probes, and physical step recovery.
- Removed inner-product checked scans and duplicate normal-CG assembly.
  Native data/normal residual estimates have explicit physical/transformed
  scopes; no forced exit J/J* evaluation. Normal-CG preparation/count test
  verifies one setup pair plus one iteration pair.
- Focused solvers, Bent, relinearized Born and reduced-model suite: 131 passed
  (2.72 s). Complex 8x3 Jacobian plus real 5x3 penalty, weighted data, nonzero
  prior, scales 1e-6/1/1e6, all three solvers and both preconditioners agree with
  independent dense least squares to 1e-9 relative / 1e-23 absolute tolerance.
- Next: F09-F10 shared GN, two explicit trial policies and accepted work state.

## Final taskbook F09

- Replaced the separate Bent and nonlinear Born loops with `gauss_newton`, two
  named problem preparers and two explicit trial generators. Kept the Bent
  coarse basis, water prior, ROI, physical update norm and fine-grid penalty;
  kept Born's initial/explicit prior and per-update speed constraints.
- Regularization has one prepared forward/adjoint implementation. Accepted
  trial linearizations are reused, not recomputed at the next iteration.
- Validation: nonlinear, Bent, nonlinear Born and reduced-model tests: 52 passed.
  A first invocation referenced a nonexistent test filename; it ran no tests.
- Next: F10 complete accepted-state ownership and boundary budget handling.
