# Evaluation Acceptance Protocol

## Required Per-Case Artifacts

Each algorithm/case run must write:

- `result.h5`
- `metrics.json`
- `metadata.yaml`
- `preview.png` for successful image-producing runs
- `failure_report.md` for failed or skipped runs

The evaluator records `artifacts_complete` and includes missing artifacts in `fail_reasons`.

## Status Semantics

- `success`: algorithm completed and wrote result artifacts.
- `failed`: algorithm attempted execution and encountered a schema, data, numerical, convergence, or dependency error.
- `skipped`: algorithm is optional or unavailable in the current environment, with a clear `failure_reason`.

Non-success records fail benchmark acceptance unless the protocol explicitly treats them as expected in a future extension.

## Protocol Fields

Benchmark YAML supports:

```yaml
min_cases: 1
min_records: 2
expected_statuses:
  - success
require_algorithm_case_matrix: true
required_metrics:
  straight_sart:
    - data_residual_norm
    - rmse
thresholds:
  straight_sart:
    rmse: 100.0
    data_residual_norm: 1.0e-4
minimums:
  straight_sart:
    water_relative_rmse_improvement: 0.0
```

Resolution policy:

- 64x64 converted cases are interface smoke diagnostics only.
- 256x256 converted cases are the minimum visual quality-comparison target for
  OpenBreastUS/NBPslice2D map-surrogate panels.
- Benchmark reports must retain `case_type` and `benchmark_type` so quality
  comparison is not confused with measured wavefield inversion.

`min_cases` and `min_records` are run-level evidence checks. `expected_statuses`
lists acceptable result states, and `require_algorithm_case_matrix` requires
every expected algorithm to have a record for every observed case. Expected
algorithms are inferred from the suite `algorithms` list unless
`expected_algorithms` is supplied explicitly.

`required_metrics` names metrics that must be present and finite. `thresholds`
are maximum allowed values. `minimums` are minimum required values. These fields
can be either global mappings or algorithm-specific mappings keyed by algorithm
name.

## Required Metrics

Sound-speed algorithms should report:

- ROI RMSE, MAE, NRMSE;
- PSNR and SSIM where valid;
- water/reference baseline RMSE improvement;
- data residual norm when a forward model is available;
- coverage nonzero fraction, valid ray fraction, row/column norm ranges, ring artifact index, and coverage/error correlation for straight-ray algorithms;
- runtime and peak memory.

Straight-ray, Bent-ray, and r-Wave quality protocols use the same standard
image and forward-model residual fields:

- `rmse`, `mae`, `nrmse`, `psnr`, `ssim`;
- `data_residual_norm`, `data_relative_residual`, and
  `data_residual_reduction`;
- `water_relative_rmse_improvement`;
- `coverage_nonzero_fraction`, `valid_ray_fraction`, and
  `ring_artifact_index`.

`bent_ray_gn` and `rwave_adapter` are runnable native v0.1 baselines by
default. Their MATLAB paths are optional dependency checks and may return
`skipped` only when the protocol explicitly exercises the external backend.

Attenuation algorithms should report:

- attenuation RMSE, MAE, NRMSE on synthetic or labeled cases;
- data residual norm;
- runtime and peak memory.

## k-Wave FWI Metrics

`fwi_kwave_adapter` has two distinct protocol roles:

- `fwi_kwave_adapter_smoke` is an ingestion smoke test. It verifies an existing
  MATLAB/k-Wave result can be loaded into the standard artifact layout. It must
  not use `loss_decreased`, water-baseline improvement, or wrapper-case RMSE as
  quality gates.
- `fwi_kwave_full_pipeline_smoke` is the A100 quality smoke path. It requires
  the external result to be loaded, records selected/final/best iteration
  diagnostics, and judges FWI against the external k-Wave ground truth and
  native scalar metrics.

FWI quality acceptance should use:

- `kwave_gt_rmse` and `kwave_gt_ssim`;
- `kwave_gt_init_rmse`;
- `kwave_gt_selected_relative_rmse_improvement`;
- `kwave_native_psnr` and `kwave_native_ssim`;
- `final_iteration_rmse` as a diagnostic final-iteration record.

Generic wrapper metrics (`rmse`, `ssim`, `water_relative_rmse_improvement`) are
still written for common CSV and image-panel tooling, but they are diagnostic
for k-Wave FWI. They must not be treated as the primary pass/fail evidence.

`LOSS_ITER` is also diagnostic. A useful FWI schedule can have nonmonotonic
waveform loss, so `loss_decreased` is not a valid full-pipeline acceptance
gate.

## Reports

`usct eval` and `usct bench` write:

- `benchmark_summary.csv`
- `benchmark_run_checks.json`
- `benchmark_report.md`

The CSV includes pass/fail booleans, pass reasons, fail reasons, runtime, peak memory, artifact completeness, and failure-report presence.
It also carries `case_type` and `benchmark_type` when the case metadata records
them, so synthetic-oracle, speed-map-surrogate, and wavefield-derived runs are
not silently merged.
The run-check JSON records protocol-level failures such as missing algorithms,
too few records, too few cases, or an incomplete algorithm/case matrix. CLI
commands return non-zero if any per-case record fails or if run-level checks
fail.
