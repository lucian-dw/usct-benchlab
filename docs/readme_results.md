# README reconstruction panels

[English README](../README.md#example-results) · [中文 README](../README.zh-CN.md#重建表现)

## Scope / 展示范围

Four panels use the preserved `readme_high_frequency_reference_20260910` study:
OpenBreastUS HET/FIB/FAT/EXD (test001/201/501/701), and NBPslice2D A/B/C/D
(A540024479, B530113966, C520105117, D510022534). These are the eight prescribed
examples, not a best-case selection. The current update only renders stored final
reconstructions; it does not rerun inversion or assert that all images were
produced by the current main commit.

本轮仅重绘已有最终结果，不重新反演、不按 GT 选择最好迭代。图中 FWI 是保存的
旧高频结果，不是新 `fwi_wust` 的八样本验证。虽然每行物理 GT 一致，但这不是相同
训练划分、频率安排与计算预算下的统一排名。

## Default configs are not the figure recipe

“默认能运行”与“默认能复现 README 主图”是不同承诺。目前只提供前者，且运行仍需
满足相应输入和环境要求。主图是历史实验的最终结果重绘，不是当前默认配置的效果承诺。

The current repository YAML examples and the preserved study differ. For example,
the OpenBreastUS HET k-Wave configs contain the following differences:

| Method / setting | Current example YAML | Saved figure experiment |
|---|---|---|
| CGLS iteration limit | 30 | 1200 |
| SIRT full-sweep limit | 50 | 1200 |
| SART full-sweep limit | 20 | 400 |
| Bent outer / inner limits | 4 / 24 | 20 / 100 |
| Bent regularization amplitude | 0.02 | 0.00003 |
| Bent initialization | Typed default: configured reference | CGLS initialization, 400 setup iterations |
| Nonlinear Full-Green outer / inner limits | 3 / 20 | 12 / 60 |
| Nonlinear Full-Green initialization | Typed default: configured background | Phase-CGLS initialization, 400 setup iterations |

These are maximum budgets, not necessarily executed iteration counts; other rules
can stop earlier. Increasing iteration limits alone does not reproduce the study.
The saved native runs also specify explicit stopping tolerances, held-out channels,
ROI and case-specific inputs. Image metrics are evaluated after physical-coordinate
interpolation, not simply read from the native-grid preview.

There are also three distinct meanings of “default”: the typed Python model,
the repository example YAML, and the Agent-resolved configuration after budget
caps/trusted overrides. They need not be identical. For instance the typed CGLS
default regularizer is identity with zero strength, whereas `cgls.yaml` explicitly
selects Laplacian with amplitude 0.02. An Agent call that only chooses Laplacian
does not implicitly load that YAML's advanced regularization strength.

For a reproduction mismatch, compare one case first and exchange:

1. Repository commit, algorithm id/variant, command and input-case SHA-256.
2. Actual resolved parameters, initialization, run controls, deployment caps and
   valid/held-out masks, rather than only the submitted Agent fields.
3. `metadata.yaml`, `metrics.json`, stopping records and the reconstruction preview.
4. Metric region, GT range and evaluation grid; a full-image SSIM cannot be compared
   directly with the tissue-only SSIM printed in these figures.

The maintainer must supply the matching observations and a migrated, validated
experiment configuration before promising fresh-inversion reproduction on current
main. That end-to-end reproduction has not been performed as part of this README
refresh. Do not assume that a discrepancy is a consumer-adapter bug, or loosen
Agent parameter permissions to work around it. Advanced settings belong in a
trusted, explicitly recorded experiment profile.

## Inputs and models

| Panel tier | CGLS / SIRT / SART | Native Bent | Ray-Born | FWI reference |
|---|---|---|---|---|
| `matched` | Straight-ray-generated ToF | Eikonal-generated ToF | Fixed-background Full-Green Born pressure | Saved k-Wave FWI, separate input |
| `kwave_fullrate` | Bounded water-relative xcorr delays | Same extracted delays interpreted by Eikonal | Nonlinear Full-Green using complex pressure and water calibration | Saved k-Wave FWI, not a rerun on regenerated pressure |

The matched panel is **not all ToF**: Born needs pressure and FWI is a separate
reference. The finite-band xcorr delay in the k-Wave panel is not certified
geometric first-arrival time. This distinction matters particularly for Bent.

The five new methods share regenerated object/water acquisitions, native temporal
sampling, actual sampled-node coordinates and a common near-neighbor mask.
Receiver indices 7, 23, ..., 119 and reciprocal pairs were held out. Historical
FWI did not use an equivalent saved split. These held-out channels are validation,
not an independent test dataset.

| Acquisition | OpenBreastUS | NBPslice2D |
|---|---|---|
| TX / RX | 128 / 128 | 128 / 128 |
| Physical simulation grid | 492 × 492 | 512 × 512 |
| External PML | 10 pixels | 20 pixels |
| Source peak | 1 MHz | 250 kHz |
| Native inverse grid: ray / Bent | 256 × 256 | 256 × 256 |
| Native inverse grid: nonlinear Full-Green | 601 × 601 | 301 × 301 |
| Full-Green frequencies | 300, 400, 500, 600, 700, 800 kHz | 150, 200, 250, 300, 350 kHz |

Both acquisitions used a 110 mm ring radius, ±120 mm coordinate range, CFL 0.3,
constant density 1000 kg/m³ and zero attenuation. The geometry-derived update
disk had radius 105 mm, not a GT tissue mask.

Saved FWI used 21 frequencies from 300 to 800 kHz in 25 kHz steps, three updates
per frequency. In NBPslice2D that schedule extends above the source band: a listed
inversion frequency is not a guarantee of useful source energy. Regenerated
waveforms correct temporal-decimation/geometry handling and must not be called
byte-identical copies of the historical FWI acquisition.

## Budgets and interpretation

| Method | Recorded maximum work | Time safety cap |
|---|---|---|
| CGLS | 1200 iterations | 3600 s |
| SIRT | 1200 complete sweeps | 3600 s |
| SART | 400 complete sweeps, 8 subsets | 3600 s |
| Bent | 20 outer / 100 inner | 7200 s |
| Matched fixed Born | 400 linear iterations | 7200 s |
| Nonlinear Full-Green | 12 outer / 60 inner | 7200 s |

The published [CSV](assets/reconstruction/metrics.csv) preserves actual completed
iterations and stopping reasons. A successful process may terminate on budget or
stagnation; it does not establish numerical convergence. In particular the k-Wave
Bent images show severe artifacts. They are retained rather than hidden or
described as accepted image-quality results. This figure refresh makes no new
claim that observation adaptation or reconstruction quality is solved.

## Evaluation and display

- All columns are evaluated on the saved FWI reference's 801 × 801 physical grid.
  Other maps are linearly interpolated at those coordinates, not rotated to match
  GT. The display grid is not the native inversion resolution.
- Each dataset shares one gray scale based on its GT extrema. Values outside the
  scale saturate visually; metrics use the unmodified arrays.
- PSNR/SSIM in the panels use the same post-hoc tissue policy within each row.
  Tissue means the complement of boundary-connected water; water reference is
  1500 m/s and classification tolerance is 0.1 m/s. This GT mask is evaluation-only.
- PSNR/SSIM data range is the full GT range of each case, shared across that row's
  methods. SSIM uses complete valid windows (up to 7 pixels), with the documented
  fallback for tiny masks. It is not computed on display thumbnails.
- CSV also includes tissue RMSE, full-image RMSE/PSNR/SSIM and water-background
  RMSE. No per-method contrast stretching, sharpening or best-GT selection is used.
- The [manifest](assets/reconstruction/manifest.json) records source/result hashes,
  evaluation masks/policy, native shapes and relative artifact paths. Raw arrays
  remain outside Git. FWI is explicitly a stored reference, not a newly verified
  `fwi_wust` run.

## Re-render

Obtain the preserved study directory from the experiment owner, keeping its
dataset/class hierarchy, `reference.npz`, input cases, completion JSON and result
HDF5 files. Installing BenchLab alone does not download this experiment bundle.

```bash
pip install -e ".[viz]"
python scripts/render_readme_results.py \
  --root /path/to/readme_high_frequency_reference_20260910 \
  --out docs/assets/reconstruction
```

The script verifies input-case hashes, requires finite successful saved native
results, and rejects evaluation coordinates outside the reconstruction domain
(apart from floating-point endpoint roundoff). It writes four grayscale PNGs,
metrics and a manifest without launching simulations or reconstructions.

Original experiment YAML files are historical records, not promised to pass the
current strict typed interface unchanged. In particular historical fixed-Born
and CUDA controls require migration; use current algorithm discovery and typed
configs for new experiments. New production WUST runs must follow [fwi.md](fwi.md).
