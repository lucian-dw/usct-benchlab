# Refactor changelog

## Post-review C01-C03

- Bent retains completed CGLS initialization when the first GN evaluation cannot
  start; an unevaluated image does not become a selected iteration checkpoint.
- Nonlinear Born transfers its prepared initial Linearization through the forward
  object once, dropping caller references before GN. Water calibration temporaries
  are released after fitting; fixed Born retains its working operator.
- Comparison rendering selects experiment ids with `--experiments` and
  `--experiment-labels`; CSV identity comes from saved run/HDF5 metadata.
- Regression: full Python 437 passed, 1 CUDA skip; CPU Full-Green ownership
  transfer checked separately. GPU peak memory was not measured in this patch.

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
  codecs use `write_artifact`, and suites load each case once.
- Detailed traces require `diagnostics: true`. Undefined relative improvement
  is null with a status; accepted histories and work counters remain available.

## Round 3: current interfaces and numerical ownership

- `rwave_adapter`, `RayBornAlgorithm` and `algorithms/rwave.py` become `born`,
  `BornAlgorithm` and `algorithms/born.py`. The four explicit WKB/Full-Green,
  fixed/nonlinear variants remain; this is not an upstream r-Wave reproduction.
- Readers use `resample_bilinear`, `resample_labels_nearest` and
  `read_kwave_pressure_mat`. The pressure reader returns a case; its caller
  derives requested features before saving once. Time/frequency water references
  and their physical units are distinct. Born conjugates pressure, source and
  frequency water reference together from the declared Fourier contract.
- Each benchmark experiment requires a unique `id`. Numerical artifacts,
  summaries and results retain that identity and the selected variant.
- AcceptedState owns model/prediction/linearization; the stopping monitor owns
  scalar history. Selected checkpoint optimality is not the last work point's
  stationarity. GN direction proposals/Armijo and CG weight refresh/restart are
  explicit steps, with prepared regularizers and opt-in detailed diagnostics.
- Configured Born reuses its initial Linearization. Prediction and requested
  normal diagonal share a frequency loop; actual source solves and matvecs are
  counted where executed. GPU background and Jacobian share a device field cache.
- AlgorithmEntry owns parameter/budget/stopping capabilities. Python minimum is
  now 3.12, `describe-algorithm` is always JSON, and unknown top-level options fail.
- WUST file ingestion prepares observations once; reconstruction consumes them.
  Update-mask PML and sampling Dirichlet restrictions are separate. Final full
  frequency evaluation is admitted before launch. Missing values are producer
  JSON null, not reader compatibility rewrites.

See [validation](docs/validation.md) for this round's executed local and A100 gates.
