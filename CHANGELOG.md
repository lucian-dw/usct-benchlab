# Refactor changelog

## Current public workflow

- One typed configuration, static registry and `run / suite / evaluate` API.
- Separate scientific parameters, run controls, host limits and deployment.
- Save numerical arrays and run state before optional evaluation or preview.
- One straight-ray CSR kernel, one FMM kernel, shared regularization and linear
  subproblem solvers; independent numerical references remain in the tests.
- Explicit physical resampling and selected OpenBreastUS files/slices; one
  property-map surrogate builder and one complex source calibration core.
- WUST protocol v2: one external reconstruction invocation, WUST-owned physical
  preparation and Hz schedule, no source/build identity admission.
- Historical experiments, panels and source snapshots live in the sibling
  research archive. Original raw data and runs are preserved.

## Migration from the previous API

| Previous use | Current use |
| --- | --- |
| Algorithm-specific dictionaries and repeated alias parsing | `AlgorithmConfig` prepared once; `config.parameters` is a concrete typed object |
| `stopping`, `BudgetCaps`, iteration aliases | `run_controls`; independent `host_limits` intersect requested budgets |
| `damping` as the quadratic penalty coefficient | `regularization_lambda = sqrt(damping)`; objective uses `regularization_lambda**2` |
| Born backend flags and variant aliases | One `variant`: `wkb_fixed`, `wkb_nonlinear`, `full_green_fixed`, or `full_green_nonlinear` |
| Runtime paths in scientific parameters | `deployment.runtime_root`, `matlab_executable`, `backend`, `scratch_root` |
| Old WUST manifests and multiple prepare/reconstruct launches | Protocol 2, one reconstruct call with frequency input; update both repositories together |
| `metadata.yaml` and inferred reporting states | `run.json`, `index.json`, independent `metrics.json`, optional `preview.png` |
| Automatically selected OpenBreastUS class/index groups | Explicit files, sample indices, axes, units and spacing |
| `requirements.txt` and duplicated extras | `pip install -e ".[dev,viz]"`; dependencies in pyproject.toml |

Unknown names fail rather than being silently mapped. YAML files and all current
examples use the new fields. No universal `update_rtol` or calibrated presets
were introduced. SSIM without a valid local window is unavailable; perfect PSNR
is JSON null with an explicit perfect-reconstruction status.

## Numerical corrections

Homogeneous Eikonal object/water times are subtracted before adding the geometric
reference, avoiding a roundoff-only failure at an exactly constant model.
The WUST native launcher now applies each TX's local pressure-source scale,
instead of reusing the first TX scale in a heterogeneous source region.
The final numerical corrections below intentionally change the affected defective behavior.

See [usage](docs/usage.md), [parameters](docs/parameter_contract.md) and the
[execution record](docs/validation.md) for actual validation and limits.

## Final numerical and interface consolidation

- Fixed Born now uses bounded projected CG with its complex pressure offset.
  Real model gradients use Re(J*Wr); constrained active-set and Huber refresh
  behavior share the straight-CGLS implementation.
- One GN driver preserves Bent coarse-space/water priors and Born model priors,
  with distinct feasible trial formulas. Accepted Linearizations are reused.
- Real/complex augmented solves declare data dtype, use ordinary scalar
  normalization and optional column_rms scaling; no exit-time J/J* recomputation.
- Corrected anisotropic Green distances and phase-unwrap support filtering.
- `irls_iterations` migrates to `irls_max_inner_steps`; explicit speed-volume
  `dataset_name` replaces largest-array guessing. Removed unused
  `boundary_band_pixels` and WUST `model_reference_speed_mps`.
- `prepare_config` owns runtime preparation; direct algorithm calls require its
  result. Agent admission and case facts do not resolve host deployment/policy.
- Dataset readers own their formats, pressure units share one contract,
  codecs use `publish_result`, and suites load each case once.
- Detailed traces require `diagnostics: true`. Undefined relative improvement
  is null with a status; accepted histories and work counters remain available.
