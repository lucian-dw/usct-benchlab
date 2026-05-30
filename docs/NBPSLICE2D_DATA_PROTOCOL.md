# NBPslice2D data protocol

`NBPslices2D` is treated as a numerical phantom map dataset, not measured USCT
RF data. The A100 reference archive is expected at:

```text
$HOME/USCT_kwave/openbreastus_diffusion/data/openbreastus_speed_crop400/NBPslices2D.zip
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
- `fwi_tiny`

External MATLAB/k-Wave adapters are intentionally not run by this smoke suite
unless a matching external result or launch contract is added later.
