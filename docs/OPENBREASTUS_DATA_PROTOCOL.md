# OpenBreastUS Data Protocol

## Purpose

OpenBreastUS data must be inspected before any path, density class, split, or file schema is assumed. The A100 workspace is the authoritative runtime location.

## Expected Paths

```bash
export USCT_WORKSPACE=$HOME/usct-benchlab
export USCT_DATA_ROOT=$USCT_WORKSPACE/data/openbreastus
export USCT_SAMPLE_ROOT=$USCT_WORKSPACE/data/openbreastus_sample
export USCT_RUN_ROOT=$USCT_WORKSPACE/runs/usctbench_runs
```

Do not use `/data/...` and do not require sudo.

## Inspection

```bash
usct data inspect-openbreastus --root "$USCT_DATA_ROOT" --out "$USCT_RUN_ROOT/openbreastus_index.json"
```

The inspector writes:

- root path;
- file count;
- conservative case records;
- inferred density class when possible;
- inferred file roles;
- frequency hints from filenames;
- dataset schema summaries for supported HDF5/MAT/NPY/NPZ files, including shape, dtype, and dataset names where available;
- per-case capabilities such as sound-speed maps, attenuation maps, wavefields, reference/water data, geometry, masks, and supported conversion modes;
- per-case limitations when the evidence is speed-only, lacks reference data, lacks geometry, or has no supported automatic converter;
- warnings when schema evidence is weak.

## Smoke Subset

```bash
usct data make-smoke --root "$USCT_DATA_ROOT" --out "$USCT_SAMPLE_ROOT" --cases-per-density 1
```

Smoke conversion defaults to 64x64 and 32 synthetic transducers. It is an
interface and numerical diagnostic, not a visual quality comparison.

The smoke builder writes:

- `openbreastus_index.json`
- `openbreastus_smoke_manifest.json`
- `schema_inspection_report.md`
- source symlinks under `sources/`
- converted standard cases under `cases/` when supported
- a `case_capability_summary` in the manifest with convertible cases, conversion mode counts, and case limitations

For speed-map-only MATLAB v7.3 mirrors such as `breast_train_speed.mat`, conversion creates downsampled standard `USCTCase` HDF5 files with straight-ray surrogate features. Metadata records the speed-only limitation, spacing assumption, and the fact that `log_amp` is a zero surrogate.

Converted speed-map cases must include provenance metadata:

- `conversion: speed_map_to_straight_ray_surrogate`
- `feature_provenance: surrogate_delta_tof_from_ground_truth_sound_speed`
- `measurement_limitations` listing the missing measured wavefield, generated straight-ray features, zero `log_amp` surrogate, and synthetic geometry assumption

For compact k-Wave simulation MAT files with datasets `C`, `atten`, `full_dataset`, and `transducerPositionsXY`, conversion creates a standard `USCTCase` with both sound-speed and attenuation ground truth. The smoke HDF5 stores straight-ray delay features from `C` and attenuation line-integral features from the simulated attenuation map. Metadata records that this is simulated attenuation evidence, not raw measured OpenBreastUS RF data, and the source channel tensor is not copied.

Converted cases include explicit type metadata so reports do not mix evidence
levels:

- `synthetic_oracle`: fully synthetic fixture generated from known ground truth.
- `openbreastus_speedmap_surrogate`: OpenBreastUS/NBPslice2D map converted to
  straight-ray feature surrogates; useful for benchmark plumbing and classical
  tomography diagnostics, but not a measured waveform benchmark.
- `openbreastus_wavefield`: k-Wave simulation source with wavefield evidence in
  the source MAT; the current smoke HDF5 still stores feature-domain line
  integrals rather than raw waveforms.

Sound-speed comparison panels should use a grayscale colormap and include GT
plus traditional sound-speed algorithms only. Use
`scripts/render_class_comparison_panels.py` for reproducible class-comparison
figures.

## 256x256 Quality Comparison

Use 256x256 converted cases for visual quality comparison:

```bash
export USCT_QUALITY_SAMPLE_ROOT=$USCT_WORKSPACE/data/openbreastus_quality_256
export USCT_QUALITY_RUN_ROOT=$USCT_WORKSPACE/runs/usctbench_runs
usct data make-quality \
  --root "$USCT_DATA_ROOT" \
  --out "$USCT_QUALITY_SAMPLE_ROOT" \
  --cases-per-density 1 \
  --converted-shape 256 \
  --n-transducers 128
usct bench --suite configs/benchmarks/openbreastus_quality.yaml
```

The convenience script `scripts/run_openbreastus_quality.sh` runs both steps and
then writes a reproducible grayscale GT+algorithm panel under
`<run_root>/comparison_artifacts/openbreastus_quality_256_sound_speed_gray.png`
with adjacent `.summary.csv` and `.manifest.json` files.

The converter rescales the full speed-map field of view to the target shape,
rather than center-cropping 480x480 maps down to 256x256. The benchmark remains
`openbreastus_speedmap_surrogate` unless the case metadata explicitly says
`openbreastus_wavefield`; it should not be described as measured RF inversion.

For canonical OpenBreastUS speed-map volumes such as `breast_train_speed.mat`,
the quality converter treats the 3-D volume as four contiguous class blocks
when no explicit per-sample labels are present. With the current A100
`breast_train: shape=(480, 480, 7200)` mirror, `--cases-per-density 1` selects
source indices `0`, `1800`, `3600`, and `5400`, and records
`openbreastus_class_id` plus `density_class=openbreastus_class_<id>` in each
converted case. This keeps quality panels class-balanced while preserving the
speed-map-surrogate limitation.

Converted k-Wave cases must include provenance metadata:

- `conversion: kwave_channel_mat_to_feature_case`
- `feature_provenance: surrogate_delta_tof_from_sound_speed_and_attenuation_line_integral_from_simulated_ground_truth`
- `attenuation_evidence: simulated_ground_truth_line_integral`
- `has_simulated_attenuation: true`
- `measurement_limitations` listing that the source is a k-Wave simulation and that features are generated line integrals

## Current A100 Evidence

As of the current branch, the configured A100 data root contains:

```text
$HOME/usct-benchlab/data/openbreastus/breast_train_speed.mat
```

It is a MATLAB v7.3/HDF5 file with dataset:

```text
breast_train: shape=(480, 480, 7200), dtype=float32
```

The current smoke benchmark uses a converted speed-map case, not real RF/wavefield measurements.

A separate local A100 tree currently contains k-Wave USCT simulation MAT files under:

```text
$HOME/USCT_kwave/Simulations/datasets/kWave_train_6602.mat
```

Those files are not committed and are not automatically assumed to be part of `$USCT_DATA_ROOT`. To use one for the smoke benchmark, link or copy it into the ignored A100 data root before running `make-smoke`, then regenerate the smoke manifest. The smoke selector prefers attenuation-capable k-Wave cases over speed-only cases within the same inferred density class.

## Rules

- Do not commit dataset files, HDF5 cases, generated previews, run outputs, or symlinks under ignored data/run directories.
- If the local tree changes, regenerate the index and schema report before updating loaders.
- If measured wavefield/RF/reference data becomes available, add a schema-specific converter instead of extending the speed-only surrogate path silently.
