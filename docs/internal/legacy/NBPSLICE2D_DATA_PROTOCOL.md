# NBPslice2D data protocol

`NBPslices2D` is treated as a numerical phantom map dataset, not measured USCT
RF data. The A100 reference archive is expected at:

```text
/path/to/NBPslices2D.zip
```

The public Illinois Data Bank description records 1,089 HDF5 `.mat` slices,
with filenames `{type}{subject_id}.mat`. The leading type is the ACR BI-RADS
composition class:

- `A`: almost entirely fatty
- `B`: scattered fibroglandular density
- `C`: heterogeneously dense
- `D`: extremely dense

Each file contains `sos`, `den`, `att`, `y`, `label`, and `type`. The converter
uses the published units:

- `sos`: mm/us, converted to m/s by multiplying by `1000`
- `att`: dB/(MHz^y mm), converted to Np/m at the configured frequency
- `den`: kg/mm^3, recorded in metadata only
- `label`: tissue labels, used to derive the ROI mask
- pixel spacing: 0.1 mm

## Conversion

The standard smoke conversion is:

```bash
usct data make-nbp-smoke \
  --zip "$USCT_NBP_ZIP_PATH" \
  --out "$USCT_NBP_SAMPLE_ROOT" \
  --cases-per-type 1 \
  --converted-shape 64 \
  --n-transducers 32
```

This 64x64 path is for interface diagnostics and fast numerical checks. It is
not the quality-comparison path.

This writes:

```text
$USCT_NBP_SAMPLE_ROOT/nbpslice2d_index.json
$USCT_NBP_SAMPLE_ROOT/nbpslice2d_smoke_manifest.json
$USCT_NBP_SAMPLE_ROOT/cases/*.h5
```

Converted cases use the shared `USCTCase` schema. Because the archive contains
property maps rather than acquired USCT channels, the measurement fields are
surrogates:

- `delta_tof_s`: straight-ray line integral of slowness relative to the
  configured reference sound speed
- `log_amp`: negative straight-ray line integral of attenuation in Np/m
- `valid_mask`: all transmitter/receiver pairs except self-pairs

These assumptions are recorded in each case metadata under `conversion`,
`feature_provenance`, and `measurement_limitations`.

Before downsampling and inversion, the converter fits the image-domain field of
view to the breast label mask. It crops a square region around `label > 0`,
leaving margin so the breast occupies about 72% of the reconstruction width.
The original source shape, crop box, and effective meter spacing are recorded in
`metadata.roi_fit` and `metadata.effective_spacing_m`. This avoids wasting most
of the reconstruction grid on blank background for small NBPslice2D samples.

For visual quality comparison, generate 256x256 cases:

```bash
export USCT_NBP_QUALITY_SAMPLE_ROOT=$USCT_WORKSPACE/data/nbpslice2d_quality_256
export USCT_NBP_QUALITY_RUN_ROOT=$USCT_WORKSPACE/runs/usctbench_runs
usct data make-nbp-quality \
  --zip "$USCT_NBP_ZIP_PATH" \
  --out "$USCT_NBP_QUALITY_SAMPLE_ROOT" \
  --cases-per-type 1 \
  --converted-shape 256 \
  --n-transducers 128
usct bench --suite configs/benchmarks/nbpslice2d_quality.yaml
```

The convenience script `scripts/run_nbpslice2d_quality.sh` runs both steps and
then writes a reproducible grayscale GT+algorithm panel under
`<run_root>/comparison_artifacts/nbpslice2d_quality_256_sound_speed_gray.png`
with adjacent `.summary.csv` and `.manifest.json` files.

The 256x256 path uses the same ROI crop, then rescales that fitted field of view
to the benchmark grid. This keeps GT detail visible while keeping the benchmark
explicitly marked as a map-surrogate, not measured RF reconstruction.

## Smoke benchmark

The script:

```bash
scripts/run_nbpslice2d_smoke.sh
```

creates the smoke subset and runs `configs/benchmarks/nbpslice2d_smoke.yaml`.
The smoke suite covers the local runnable algorithm library flow:

- `straight_sart`
- `straight_sirt`
- `straight_cgls`
- `attenuation_sirt`

`fwi_tiny` is intentionally not part of this smoke comparison. It is a
synthetic proof-of-life FWI check, and on map-surrogate NBPslice2D samples it
can produce visually uninformative near-constant panels. External MATLAB/k-Wave
adapters are also intentionally not run by this smoke suite unless a matching
external result or launch contract is added later.

Sound-speed comparison panels should use a grayscale colormap and include GT
plus traditional sound-speed algorithms only. Use
`scripts/render_class_comparison_panels.py` for reproducible class-comparison
figures.
# NBPslice2D measurement status

The 2025 `NBPslices2D.zip` integration currently treats the dataset as property-map-only: `sos`, `den`, `att`, `y`, and `label`. The converter stores GT sound speed/attenuation and creates solver-sanity features from the property maps, but those features are `speedmap_travel_time_surrogate`.

If no built-in wavefield is present, NBPslice2D should be used as a property-map library for `self_simulated_kwave_wavefield` benchmarks. The related 2021 measurement dataset needs separate inspection before any `external_measurement` or precomputed-wavefield support is claimed.
