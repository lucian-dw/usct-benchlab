# k-Wave FWI Success Notes

## Scope

This historical experiment note records the successful v0.1 k-Wave FWI path
without vendoring data, MATLAB code, generated RF fields, or run outputs. The
adapter remains a standard `USCTCase -> ReconstructionResult` algorithm, while
the full-wave simulation and inversion live in an external A100 `USCT_kwave`
tree configured through `$USCT_KWAVE_ROOT`.

## Code Map

- `src/usctbench/algorithms/fwi/kwave_adapter.py` reads external MATLAB v7.3
  FWI results and optionally launches the A100 full pipeline.
- `configs/algorithms/fwi_kwave_full_pipeline.yaml` stores the current
  bulk-support pure-FWI launch parameters.
- `configs/benchmarks/fwi_kwave_full_pipeline_smoke.yaml` defines the FWI smoke
  acceptance gates.
- `scripts/run_fwi_kwave_full_pipeline_smoke.sh` creates the 256x256 benchmark
  wrapper case from the existing `preproc_crop300_tla250` k-Wave MAT, launches
  or ingests the external result, renders visual artifacts, and evaluates the
  run.
- `scripts/render_kwave_fwi_smoke_outputs.py` renders a `contact_sheet.png`
  with GT, selected/final/best reconstructions, aligned error panels, plus the
  individual reconstruction, error, loss, gradient, and metadata artifacts.
- The full-pipeline smoke script runs FWI only. Straight-ray, bent-ray
  surrogate, and rWave surrogate comparisons belong in
  `configs/benchmarks/travel_time_surrogate_main.yaml`, not in the k-Wave FWI
  script.

## Current Bulk-Support Pure-FWI Settings

The current A100 smoke path is `invert_existing_dataset` on OpenBreastUS
preprocessed k-Wave datasets such as test201:

- wrapper grid: 256x256;
- external array mode: `full128`;
- external data: `preproc_crop300_tla250`, `atten_bkgnd=0`, `sos2atten=0`,
  CUDA binary RF data;
- external frequency continuation: `0.3:0.025:0.8 MHz`;
- sound-speed iterations: 3 per frequency;
- attenuation inversion: disabled;
- reconstruction grid: `0.3 mm`;
- bulk-support warm start: enabled;
- warm-start module:
  `openbreastus_diffusion.kwave_dps.make_bulk_support_init`;
- warm-start settings: Hilbert arrival picker, min confidence `1.5`,
  residual clip `5 us`, residual percentile `95`, max rays `8000`,
  smooth sigma `10 mm`, support backprojection, support percentile `45`,
  support/support-alpha sigmas `8 mm`, support dilation `4 mm`,
  bulk update scale `1.0`, min path `20 mm`, bulk statistic `median`;
- FWI sign convention: `sign_conv=-1`;
- update damping: `0.25`;
- velocity bounds: `[1408.692, 1595.1279] m/s`;
- per-step update clamp: `12 m/s`;
- background attenuation: `0`;
- raw gradient dumps: disabled by default.

The preferred sample201 dataset is configured through `USCT_KWAVE_DATASET_PATH`.
For the A100 experiment, it pointed to:

```text
$USCT_KWAVE_ROOT/openbreastus_diffusion/kwave_dps/outputs/dps_preproc_test201_latestckpt/datasets/kWave_openbreast_test201_full128_scale1p000_preproc_crop300_tla250_ab0_s2a0_cuda_binary.mat
```

Reference pure-FWI inside metrics from
`$USCT_KWAVE_ROOT/openbreastus_diffusion/kwave_dps/docs/experiments/2026-06-02_bulk_support_dps_mainline.md`:

```text
sample001: PSNR=26.1998 SSIM=0.7887 corr=0.9368 HP-corr=0.8869
sample201: PSNR=26.4539 SSIM=0.8315 corr=0.9520 HP-corr=0.9240
sample501: PSNR=26.1279 SSIM=0.7936 corr=0.9327 HP-corr=0.8941
sample701: PSNR=29.3894 SSIM=0.8216 corr=0.8414 HP-corr=0.8092
```

The old RF texture/travel-time warm start and `--update-scale -1` are no longer
the default. They are retained only as ablation/debug paths.

The current adapter path remains pure FWI and does not invoke diffusion/DPS.

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
`$USCT_KWAVE_ROOT`, datasets, RF fields, and `.mat` results must remain outside
Git.
