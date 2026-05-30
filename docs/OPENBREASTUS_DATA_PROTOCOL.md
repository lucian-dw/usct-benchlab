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

## Rules

- Do not commit dataset files, HDF5 cases, generated previews, run outputs, or symlinks under ignored data/run directories.
- If the local tree changes, regenerate the index and schema report before updating loaders.
- If measured wavefield/RF/reference data becomes available, add a schema-specific converter instead of extending the speed-only surrogate path silently.
