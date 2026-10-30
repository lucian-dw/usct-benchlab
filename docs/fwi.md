# WUST full-wave inversion

`fwi_wust` calls the independent maintained WUST runtime. One v2 reconstruction
request starts one MATLAB process, prepares supplied observations, resolves the
physical frequency schedule and runs sound-speed FWI. It never generates
observations from ground truth. MATLAB/GPU are not normal Python CI dependencies.

## Deployment

Install `lucian-dw/WaveformInversionUST` with protocol **2**, NumPy and h5py in the
Python environment, and MATLAB on the host. Build WUST's CUDA MEX for GPU use.
Select CPU or GPU explicitly; there is no automatic backend fallback.

```bash
export WUST_RUNTIME_ROOT=/path/to/WaveformInversionUST
python "$WUST_RUNTIME_ROOT/Runtime/python/wust_runtime.py" describe
usct run fwi_wust --case /path/to/frequency_case.h5 \
  --config configs/algorithms/fwi_wust.yaml --backend cpu --out runs/single_fwi
```

The example profile is not experimentally calibrated. Resolve bounds and PML for
the actual grid. Deployment owns the runtime root, MATLAB executable, device and
scratch directory; these are not autonomous Agent parameters. Version discovery
is static. No source hash, build manifest, approved revision or dirty-tree gate
is required.

## Input and preparation

- `freq_data[frequency,tx,rx]`: complex **total** pressure.
- `frequencies_hz`: positive unique frequencies, with an explicit binary
  `valid_mask[tx,rx]` or `[frequency,tx,rx]`. Valid zero observations remain data.
- Image grid `[y,x]`, metric spacing `[dy,dx]`, pixel-edge origin. BenchLab swaps
  position columns from its `[y,x]` to WUST's `[x,y]` exactly once.
- `pressure_contract`: `fourier_sign`, `real_pressure`,
  `pressure_type: total_pressure`, `data_units`, `spectrum_normalization`.

WUST owns frequency sorting, mask application, Fourier conjugation, snapping and
MATLAB indices. A +1 transform can be conjugated only for declared real RF.
ToF, scattered pressure and pressure ratios are not total-pressure FWI inputs.
WUST fits a complex source scale per TX and frequency; it does not require a
Born source spectrum or water reference.

Initialization is a declared reference speed, explicit scalar or expert map.
Maps must match grid and bounds without resizing or clipping. The default update
mask excludes PML plus one maximum spacing from the edges; an expert may supply
an explicit mask. WUST rejects a mask entering its PML/boundary. GT is absent
from reconstruction and reserved for evaluation.

## Schedule and stopping

`frequency_schedule_hz: null` means each sorted available frequency once.
Repeated Hz entries mean repeated updates. An explicit empty plan or zero
iteration budget makes no updates. MATLAB maps exact physical Hz values to
indices, then truncates the full plan by the effective `max_updates` budget.

One update optimizes full slowness `s=1/c`, measured in s/m, and records
`norm(s_new-s_old)/norm(s_old)` over the update mask. This diagnostic does not
activate a convergence threshold. Schedule completion is completion, not
convergence. Truncation reports `max_iterations`. The mandatory elapsed-time
deadline covers preparation, launch, solve and result reading; timeout kills the
owned process group and reports time-budget exhaustion without a completed image. Forward/adjoint
caps and non-null `update_rtol` are unsupported.

## Outputs and validation

The authoritative `c_mps[y,x]` is float64. Each record's loss/residual is evaluated
**before** its frequency update; unavailable final-model residual remains null.
History is optional at the runtime level. Results record actual software version,
backend, precision, schedule, consumed updates and logs, without array digests or
whole-array write/readback passes.

The one cross-repository acceptance command shares WUST's coherent fixture and
checks a non-square grid, masks, Fourier sign, physical schedule and loss decrease:

```bash
python scripts/validate_wust_integration.py \
  --runtime "$WUST_RUNTIME_ROOT" --matlab /path/to/matlab \
  --backend cpu --out /absolute/new/acceptance-directory
```

It runs fixture generation separately, then one reconstruction. Choose `--backend
gpu` for the GPU gate. Small CPU/GPU acceptance proves this declared numerical
path, not clinical validity or deployment-scale performance.
Use `--fixture /absolute/existing/fixture-directory` to reuse the same coherent
observations and initial map across backends instead of regenerating the fixture.
