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
thresholds:
  rmse: 100.0
  data_residual_norm: 1.0e-4
minimums:
  water_relative_rmse_improvement: 0.0
```

`thresholds` are maximum allowed values. `minimums` are minimum required values.

## Required Metrics

Sound-speed algorithms should report:

- ROI RMSE, MAE, NRMSE;
- PSNR and SSIM where valid;
- water/reference baseline RMSE improvement;
- data residual norm when a forward model is available;
- runtime and peak memory.

Attenuation algorithms should report:

- attenuation RMSE, MAE, NRMSE on synthetic or labeled cases;
- data residual norm;
- runtime and peak memory.

## Reports

`usct eval` and `usct bench` write:

- `benchmark_summary.csv`
- `benchmark_report.md`

The CSV includes pass/fail booleans, pass reasons, fail reasons, runtime, peak memory, artifact completeness, and failure-report presence.

