# Straight-Ray SART

## Physical Assumption

Straight-ray SART treats travel-time differences as line integrals through a perturbation in acoustic slowness:

```text
delta_tof_s = integral(1 / c(x) - 1 / c_ref) ds
```

It ignores refraction, diffraction, and waveform effects, so it is a baseline rather than a high-fidelity model.

## Input Requirements

- `USCTCase.measurement.domain = features`
- `measurement.delta_tof_s` with shape `[n_tx, n_rx]`
- `measurement.valid_mask` with the same shape when rays should be excluded
- transmitter and receiver positions in meters

## Default Settings

- iterations: `10`
- relaxation: `0.5`
- reference sound speed: `1500 m/s`
- sound-speed bounds: `[1300, 1700] m/s`

## Expected Failure Modes

- wrong sign convention in `delta_tof_s`;
- transmitter/receiver ordering mismatch;
- invalid units for positions or time delays;
- over-aggressive relaxation causing streak artifacts.

## What To Adjust First

1. Inspect the travel-time sinogram and valid mask.
2. Confirm the sign convention with a slower inclusion.
3. Lower relaxation.
4. Increase iterations gradually.

## Acceptance Tests

- homogeneous phantom returns the reference sound speed;
- straight-ray projector adjoint dot-product test passes;
- positive delay through a slower object reconstructs a slower center.

