# k-Wave FWI Success Notes

## Scope

This note records the successful v0.1 k-Wave FWI path without vendoring data,
MATLAB code, generated RF fields, or run outputs. The adapter remains a
standard `USCTCase -> ReconstructionResult` algorithm, while the full-wave
simulation and inversion live in the external A100 `USCT_kwave` tree.

## Code Map

- `src/usctbench/algorithms/fwi/kwave_adapter.py` reads external MATLAB v7.3
  FWI results and optionally launches the A100 full pipeline.
- `configs/algorithms/fwi_kwave_full_pipeline.yaml` stores the successful
  full-pipeline launch parameters.
- `configs/benchmarks/fwi_kwave_full_pipeline_smoke.yaml` defines the FWI smoke
  acceptance gates.
- `scripts/run_fwi_kwave_full_pipeline_smoke.sh` creates the 256x256 benchmark
  wrapper case, launches or ingests the external result, renders visual
  artifacts, and evaluates the run.
- `scripts/render_kwave_fwi_smoke_outputs.py` renders a `contact_sheet.png`
  with GT, selected/final/best reconstructions, aligned error panels, plus the
  individual reconstruction, error, loss, gradient, and metadata artifacts.
- The full-pipeline smoke script also runs `straight_cgls`, `straight_sirt`,
  `straight_sart`, `bent_ray_gn`, and `rwave_adapter` on the same 256x256
  wrapper case by default, then writes a horizontal cross-algorithm panel under
  `<run_root>/comparison_artifacts/`. Set
  `USCT_KWAVE_RENDER_CROSS_ALGORITHM=0` to disable this extra comparison step.

## Preserved Successful Settings

The preserved A100 smoke path is `full_pipeline_from_speed_map` on OpenBreastUS
test201:

- wrapper grid: 256x256;
- external array mode: `full128`;
- external frequency continuation: `0.3:0.025:0.8 MHz`;
- sound-speed iterations: 3 per frequency;
- attenuation inversion: disabled;
- reconstruction grid: `0.3 mm`;
- RF travel-time warm start: enabled;
- warm-start sign: `--update-scale -1`;
- FWI sign convention: `sign_conv=-1`;
- update damping: `0.25`;
- velocity bounds: `[1408.692, 1595.1279] m/s`;
- per-step update clamp: `12 m/s`;
- background attenuation: `0`;
- raw gradient dumps: disabled by default.

The successful reference run on A100 is:

```text
/home/wudalong/usct-benchlab/runs/usctbench_runs/fwi_kwave_full_pipeline_success201_dense
```

The external result file for that run is:

```text
/home/wudalong/usct-benchlab/runs/usctbench_runs/fwi_kwave_full_pipeline_success201_dense/external_kwave/results/openbreast_test201_kwave_full_WaveformInversionResults.mat
```

The current best-selected re-ingest and cross-algorithm evidence run is:

```text
/home/wudalong/usct-benchlab/runs/usctbench_runs/fwi_kwave_best_grid_reingest_2b8d7a5
```

It evaluates best/final iterations on the same 256x256 `USCTCase` grid used by
the horizontal comparison panel. The representative test201 metrics are
`selected_iteration=19`, `kwave_gt_rmse=16.85`,
`kwave_gt_selected_relative_rmse_improvement=0.343`,
`kwave_gt_ssim=0.677`, `kwave_native_psnr=22.80`, and
`kwave_native_ssim=0.705`.

## Acceptance Semantics

FWI is judged primarily against the external k-Wave result ground truth
(`C_INTERP`) and the k-Wave-native scalar metrics. The benchmark wrapper case
exists for common artifact layout and comparability, but its surrogate metrics
are diagnostics rather than the main pass/fail evidence.

Required pass/fail evidence:

- `external_result_loaded`;
- `iterations`, `selected_iteration`, `selected_loss`;
- `best_iteration`, `best_iteration_rmse`, `final_iteration_rmse`;
- `kwave_gt_rmse`, `kwave_gt_init_rmse`, `kwave_gt_ssim`;
- `kwave_gt_selected_relative_rmse_improvement >= 0.1`;
- `kwave_native_psnr >= 20`;
- `kwave_native_ssim >= 0.6`.

The full-pipeline smoke path selects the best available iteration by RMSE
against the external k-Wave `C_INTERP` grid, then records final-iteration
metrics separately. The adapter still records water/background baseline metrics,
but the FWI smoke protocol does not use them as acceptance gates. This avoids
passing or failing FWI based on an image-domain baseline that is weaker than the
external full-wave evidence.

## What Not To Delete

Do not delete the following when cleaning redundant FWI files:

- the adapter and its unit tests;
- `configs/algorithms/fwi_kwave_full_pipeline.yaml`;
- `configs/benchmarks/fwi_kwave_full_pipeline_smoke.yaml`;
- `scripts/run_fwi_kwave_full_pipeline_smoke.sh`;
- `scripts/render_kwave_fwi_smoke_outputs.py`;
- the cross-algorithm panel call in `scripts/run_fwi_kwave_full_pipeline_smoke.sh`;
- this note and the `fwi_kwave_adapter` algorithm card.

Run outputs under `runs/`, external MATLAB/k-Wave code under `external/` or
`$HOME/USCT_kwave`, datasets, RF fields, and `.mat` results must remain outside
Git.
