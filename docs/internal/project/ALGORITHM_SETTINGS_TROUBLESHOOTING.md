# Classical USCT Algorithm Settings and Troubleshooting Guide

This is the “expert notes” file for Codex. When an experiment fails, do not randomly edit code. Diagnose according to the physics and the numerical method.

## 0. Global checks before tuning any algorithm

Most USCT failures are not caused by the optimizer. Check these first:

1. **Units**
   - coordinates in meters, not millimeters;
   - frequency in Hz, not kHz or MHz;
   - sound speed in m/s;
   - slowness in s/m;
   - time delay in seconds.

2. **Geometry**
   - transmitter and receiver positions match the data ordering;
   - ring radius is correct;
   - clockwise/counterclockwise ordering is consistent;
   - self-transmit/self-receive channels are masked if invalid.

3. **Reference subtraction**
   - travel-time feature should use `tof_object - tof_water` or equivalent phase-ratio delay;
   - log-amplitude should use a stable reference amplitude;
   - wrong reference sign produces inverted contrast.

4. **Missing data**
   - use NaN or `valid_mask`, not zero, for missing rays;
   - zero-filled missing rays create artificial high-confidence measurements.

5. **ROI mask**
   - metrics should ignore outside-phantom/background regions if appropriate;
   - algorithms may reconstruct on full grid but report metrics inside ROI.

6. **Visualization**
   - always inspect sinogram/matrix features before interpreting images.

## 1. Straight-ray travel-time tomography: SART/SIRT/CGLS

### Model

Straight-ray ToF tomography approximates:

```text
delta_t(tx, rx) = ∫_ray [s(x) - s0] dl
s(x) = 1 / c(x)
```

where `s0 = 1 / c0` is the water/background slowness.

### Default setting

```yaml
grid_shape: [128, 128]
background_sound_speed_mps: 1480.0
unknown: slowness_perturbation
solver: sart
iterations: 50
relaxation: 0.5
min_sound_speed_mps: 1350.0
max_sound_speed_mps: 1700.0
smoothing_sigma_px: 0.5
use_valid_mask: true
ignore_reciprocal_duplicates: true
```

For the first smoke run, try:

- 16 or 32 transmitters;
- lowest frequency-derived phase delay;
- robustly masked phase-delay matrix;
- 20 iterations, then 50.

### Useful variants

- `SART`: robust first baseline.
- `SIRT`: smoother and sometimes more stable.
- `CGLS/LSQR`: good when the operator and adjoint are correct.
- Tikhonov/Laplacian regularized least squares: useful when rays are sparse.

### Expected symptoms and fixes

| Symptom | Likely cause | First actions |
|---|---|---|
| reconstructed inclusion has wrong sign | using `water - object` instead of `object - water`, or using speed instead of slowness | run synthetic sign test; flip sign only after confirming formula |
| image full of streaks | too few rays, missing mask wrong, relaxation too high | lower relaxation to 0.1–0.3; apply valid mask; increase iterations slowly |
| SoS outside physical range | noisy ToF or unstable update | clip SoS; reconstruct slowness then convert; add smoothing |
| no improvement over water baseline | phase-delay extraction failed or geometry mismatch | inspect sinogram; verify source/receiver ordering; test synthetic geometry |
| NaN/Inf | zero reference amplitude, division by zero, invalid missing data | add amplitude threshold; use NaN mask; robust clipping |
| very slow | explicit dense matrix too large | use ray-driven sparse operator or on-the-fly projector; subsample smoke case |

### Do not do

- Do not reconstruct sound speed linearly as if `delta_t = ∫ delta_c dl`; use slowness perturbation.
- Do not use zeros for missing ToF.
- Do not judge the algorithm without a water/background baseline comparison.

## 2. Straight-ray attenuation tomography

### Model

A simple first-order attenuation model:

```text
log_amp(tx, rx, f) = ∫_ray alpha(x, f) dl + noise
```

where:

```text
log_amp = -log((|p_object| + eps) / (|p_reference| + eps))
```

### Default setting

```yaml
solver: sirt
iterations: 50
relaxation: 0.3
robust_clip_percentile: [1, 99]
min_attenuation_np_per_m: 0.0
max_attenuation_np_per_m: 200.0
smoothing_sigma_px: 1.0
frequency_mode: single_lowest_or_average
```

### Symptoms and fixes

| Symptom | Likely cause | First actions |
|---|---|---|
| negative attenuation everywhere | ratio sign convention wrong | verify log formula and reference field |
| extreme hot pixels | low reference amplitude or bad channels | threshold reference amplitude; percentile clipping |
| no structure | amplitude dominated by source/directivity not tissue | per-transmitter normalization; remove bad channels |
| strong ring artifacts | transducer gain variation | estimate transmitter/receiver gain offsets if possible |

### Notes

This baseline is intentionally simple. It is useful for sanity and multi-parameter output, but should not be overclaimed.

## 3. Refraction-corrected / bent-ray travel-time tomography

### Model

Bent-ray tomography uses travel times but accounts for refraction through a spatially varying sound speed. Typical algorithms solve a nonlinear inverse problem with Gauss-Newton or similar updates.

### Default setting

```yaml
initial_model: smoothed_straight_ray
outer_iterations: 5
inner_iterations: 20
regularization: laplacian
tikhonov_lambda: 1e-2
line_search: true
sos_bounds_mps: [1350, 1700]
smoothing_schedule_px: [3.0, 2.0, 1.0]
```

### Initialization

Never start bent-ray GN from a noisy map. Use:

1. homogeneous water/background;
2. straight-ray reconstruction;
3. strong Gaussian smoothing;
4. clipped physical bounds.

### Symptoms and fixes

| Symptom | Likely cause | First actions |
|---|---|---|
| ray tracing fails | caustics, bad initial model, too strong contrast | smooth initial model; reduce contrast by clipping; lower resolution |
| objective increases | no line search or step too large | add backtracking line search; increase regularization |
| worse than straight-ray | feature noise or wrong regularization | use fewer trusted rays; increase smoothing; check ToF feature |
| checkerboard artifacts | under-regularized update | stronger Laplacian/TV; lower grid resolution |
| very slow | ray tracing each iteration too expensive | use smoke subset; cache paths; reduce tx/rx count |

### Rule of thumb

Bent-ray should be treated as a nonlinear correction of a straight-ray solution, not as a magic replacement for bad ToF data.

## 4. Ray-Born / Rytov / weak-scattering reconstruction

### Model

Ray-Born methods bridge ray tomography and wave inversion. They usually assume:

- a known or slowly varying background;
- high-frequency ray propagation;
- single-scattering or weak-scattering perturbation;
- complex wavefield data with reliable phase/amplitude calibration.

### Default setting

```yaml
backend: rwave_matlab_adapter
initial_background: smoothed_bent_ray_or_straight_ray
frequencies_hz: [300000, 350000, 400000]
regularization: tikhonov
lambda: 1e-2
cg_iterations: 20
valid_amplitude_percentile: 5
phase_unwrap: true
```

### Recommended workflow

1. Build a stable background using straight-ray or bent-ray ToF.
2. Use low frequency first.
3. Reject low-SNR receivers.
4. Run ray-Born update.
5. Compare both image metrics and complex/phase residual.

### Symptoms and fixes

| Symptom | Likely cause | First actions |
|---|---|---|
| update is pure noise | background too wrong or phase corrupted | improve background; use lower frequency; mask bad channels |
| high-frequency artifacts | too weak regularization | increase lambda; smooth update; use fewer frequencies |
| residual improves but image worsens | model mismatch or amplitude calibration issue | prioritize phase-only test; inspect receiver gain/directivity |
| MATLAB adapter fails | path or toolbox dependency | write exact command, save `.mat`, skip gracefully |

### Important

Do not claim ray-Born success unless the result improves over the ToF background or gives a clearly lower data residual with plausible image structure.

## 5. Tiny Full-Waveform Inversion

### Model

FWI minimizes waveform mismatch by differentiating through a wave-equation solver. In USCT it can produce high-resolution SoS/attenuation maps but is sensitive to initial model, low-frequency content, source calibration, and cycle skipping.

### v0.1 target

Only implement a tiny sound-speed-only proof-of-life.

```yaml
parameter: sound_speed_only
grid_shape: [64, 64]
initial_model: smoothed_straight_ray_or_water
frequencies_hz: [300000]
optimizer: lbfgs_or_adam
iterations: 20
learning_rate: 1e-2
sos_bounds_mps: [1350, 1700]
loss: complex_l2_or_phase_amplitude_l2
pml_size_px: 12
gradient_check: true
```

### Frequency continuation

Use low-to-high frequency:

```text
300 kHz -> 350 kHz -> 400 kHz -> ...
```

Do not start with high frequency unless the initial model is already accurate.

### Symptoms and fixes

| Symptom | Likely cause | First actions |
|---|---|---|
| loss increases immediately | gradient sign, step size, source scaling | run gradient check; reduce LR by 10x; normalize data |
| loss decreases but image wrong | cycle skipping or wrong source/geometry | lower frequency; smooth initial model; check geometry |
| ring/PML artifacts | boundary reflections | increase PML; inspect simulated wavefield; reduce grid crop issues |
| checkerboard/noisy gradients | too high frequency or poor scaling | smooth gradient; frequency continuation; clip update |
| attenuation/speed cross-talk | trying too many parameters | invert sound speed only first |
| GPU OOM | storing full wavefields | reduce tx/rx/frequency; accumulate gradients; use checkpointing |
| zero/black gradient | detached tensor, invalid source, wrong receiver sampling | test synthetic single-source; verify autograd path |

### Cycle skipping notes

If predicted and observed signals differ by more than about half a cycle, L2 FWI can lock onto the wrong minimum. First fixes:

1. lower starting frequency;
2. better initial model from ray tomography;
3. frequency differencing / low-frequency extrapolation;
4. phase or optimal-transport-style misfit later;
5. source encoding later for speed, not for first correctness.

## 6. Reflection/DAS optional baseline

If reflection data exists, add DAS as structural imaging, not quantitative SoS.

Default:

```yaml
algorithm: delay_and_sum
assumed_sound_speed_mps: 1480
apodization: hann
dynamic_receive_aperture: false
```

Failure modes:

- wrong assumed sound speed blurs focus;
- direct arrival not removed;
- wrong channel ordering;
- treating reflection image as quantitative SoS.

## 7. Tuning order cheat sheet

When something fails, adjust in this order:

1. data units and sign;
2. geometry ordering;
3. valid mask and bad channels;
4. background/reference subtraction;
5. resolution and subsampling;
6. relaxation/learning rate;
7. regularization/smoothing;
8. frequency schedule;
9. optimizer;
10. only then algorithm redesign.

## 8. Minimal default configs

### Straight SART

```yaml
algorithm: straight_sart
background_sound_speed_mps: 1480.0
iterations: 50
relaxation: 0.5
smoothing_sigma_px: 0.5
sos_bounds_mps: [1350.0, 1700.0]
feature: delta_tof_s
```

### Attenuation SIRT

```yaml
algorithm: attenuation_sirt
iterations: 50
relaxation: 0.3
robust_clip_percentile: [1, 99]
attenuation_bounds_np_per_m: [0.0, 200.0]
feature: log_amp
```

### Bent-ray GN

```yaml
algorithm: bent_ray_gn
initial_model: straight_sart_smoothed
outer_iterations: 5
inner_iterations: 20
regularization: laplacian
lambda: 1e-2
line_search: true
smoothing_sigma_px: 2.0
```

### r-Wave adapter

```yaml
algorithm: rwave_adapter
backend: matlab
initial_background: straight_sart_smoothed
frequencies_hz: [300000, 350000, 400000]
regularization_lambda: 1e-2
cg_iterations: 20
```

### Tiny FWI

```yaml
algorithm: fwi_tiny
backend: torch_or_deepwave
parameter: sound_speed_only
frequencies_hz: [300000]
iterations: 20
optimizer: adam
learning_rate: 1e-2
sos_bounds_mps: [1350, 1700]
gradient_check: true
```
