# FWI Adapter

The FWI path is exposed through `fwi_kwave_adapter`. It is designed to
ingest and report external k-Wave/FWI outputs in the same artifact format as
the classical baselines.

## Mathematical Model

The FWI objective is summarized as

$$
\min_c
\frac{1}{2}\sum_{\omega,s,r}
\left|
\hat p_s(\omega,r;c)-\hat p_{sr}^{\mathrm{obs}}(\omega)
\right|^2
+\lambda R(c).
$$

The pressure prediction $\hat p_s(\omega,r;c)$ is produced by an external
full-wave solver.

## Configure an Existing Result

Set:

```bash
export USCT_KWAVE_FWI_RESULT_PATH=/path/to/fwi_result.mat
export USCT_KWAVE_ROOT=/path/to/external/USCT_kwave
export USCT_KWAVE_PYTHON_BIN=/path/to/python
```

Then run:

```bash
usct run fwi_kwave_adapter \
  --case "$USCT_WORKSPACE/data/openbreastus_demo/cases/example_case.h5" \
  --config configs/algorithms/fwi_kwave.yaml \
  --out runs/single_fwi
```

The result MAT file must contain `VEL_ESTIM`, the final reconstructed sound
speed image. Optional fields enable richer reports:

| Field | Meaning |
| --- | --- |
| `C_INTERP` | Ground-truth sound speed used for FWI-native metrics. |
| `ATTEN_ESTIM` | Final attenuation estimate. |
| `VEL_ESTIM_ITER` | Sound-speed images over iterations. |
| `ATTEN_ESTIM_ITER` | Attenuation images over iterations. |
| `LOSS_ITER` | Per-iteration loss curve and iteration count. |
| `VEL_INIT` | Initial sound-speed model. |
| `ATTEN_INIT_USED` | Initial attenuation model. |
| `psnr_value`, `ssim_value` | Native external metrics, if saved. |
| `datasetPath` | External dataset path recorded by the FWI pipeline. |

If `run_external: true` is enabled in the config, `USCT_KWAVE_ROOT` must point
to the external solver checkout. If `USCT_KWAVE_PYTHON_BIN` is unset, the
adapter uses the current Python interpreter.

## Run the Demo Suite

```bash
export USCT_KWAVE_FWI_CASE_GLOB="$USCT_WORKSPACE/data/fwi_kwave_demo/cases/*.h5"
usct bench --suite configs/benchmarks/fwi_kwave_demo.yaml
```

## Outputs

The adapter writes the standard package artifacts:

```text
runs/single_fwi/<case_id>/result.h5
runs/single_fwi/<case_id>/metrics.json
runs/single_fwi/<case_id>/metadata.yaml
runs/single_fwi/<case_id>/preview.png
```

When ground truth is present, metrics include image quality fields such as
RMSE, SSIM, PSNR, and baseline-improvement values.

## Scope

`fwi_kwave_adapter` is an adapter route. It does not vendor a production
k-Wave or MATLAB solver into this package. External solver setup remains the
responsibility of the user environment.
