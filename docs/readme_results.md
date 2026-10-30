# Preserved reconstruction examples / 已保存的重建示例

[English README](../README.md#example-results) · [中文 README](../README.zh-CN.md#重建表现)

## Source and scope / 来源与范围

The four current panels are preserved from the main-branch study
`readme_high_frequency_reference_20260910`. No inversion or metric recomputation
is performed by this documentation update.

| Dataset | Classes | Saved sample ids |
| --- | --- | --- |
| OpenBreastUS | HET / FIB / FAT / EXD | test001 / test201 / test501 / test701 |
| NBPslice2D | A / B / C / D | A540024479 / B530113966 / C520105117 / D510022534 |

主图保留原始文件，没有裁剪、旋转、增强或更改指标。FWI 列是历史高频结果，
不是当前 `fwi_wust` 接口在这八个样本上的新验收。默认配置、当前数值实现与
当时的完整实验设置不同；不能承诺只运行默认 YAML 就复现这些图片。

## Observation models / 观测模型

| Panel | Straight CGLS/SIRT/SART | Native Bent | Born | FWI reference |
| --- | --- | --- | --- | --- |
| Model-matched | Straight-ray ToF | Eikonal ToF | Fixed Full-Green pressure | Separate saved k-Wave FWI |
| k-Wave | Water-relative bounded xcorr delays | Same extracted delays interpreted by Eikonal | Nonlinear Full-Green complex pressure with water calibration | Saved historical FWI |

The native k-Wave columns use 128 TX/128 RX regenerated object/water acquisitions.
Their native ray/Bent inversion grids are 256x256; Full-Green grids are 601x601
(OpenBreastUS) and 301x301 (NBPslice2D). The panels are displayed/evaluated on the
stored reference's 801x801 physical-coordinate grid, not the native solver grid.

Finite-band correlation delays are not certified geometric first-arrival times.
Historical FWI did not share the native columns' saved validation split or full
acquisition history. Validation channels are not an independent test dataset.
Compare model-consistent residuals within their declared domains, not ToF and
pressure residual magnitudes as one score.

## Evaluation / 评价

- Each dataset uses a shared grayscale. Maps are interpolated in physical
  coordinates, not rotated or sharpened to match GT.
- Displayed PSNR/SSIM use the same post-hoc tissue policy within each row;
  GT-derived tissue masks are evaluation-only, not reconstruction support.
- The saved CSV also records tissue/full-image/background errors, completed
  iterations and stop reasons. Budget or stagnation is not convergence.
- FWI used 21 frequencies from 300 to 800 kHz, with three updates per frequency.
  This historical schedule is not a recommended band for every acquisition.

[Saved metrics](assets/reconstruction/metrics.csv).
Raw observations and result arrays remain outside Git.

## Current plotting / 当前绘图入口

For new saved suites, use `scripts/render_examples.py --experiments ...
--experiment-labels ...`. It reads explicit experiment directories and preserves
experiment_id, algorithm and variant in its CSV. This general plotting command
does not recreate the historical data or original study-specific evaluation.

These historical pictures illustrate saved reconstructions; they are not a
substitute for the current numerical tests or a claim of clinical validation.
