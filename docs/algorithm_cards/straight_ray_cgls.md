# Straight-Ray CGLS

## Physical Assumption

Straight-ray CGLS solves the masked least-squares problem for a slowness perturbation:

```text
A delta_s = delta_tof_s
```

The final sound speed is converted from `1 / c_ref + delta_s` and clipped to configured physical bounds.

## Input Requirements

- `USCTCase.measurement.domain = features`
- `measurement.delta_tof_s` with shape `[n_tx, n_rx]`
- optional `measurement.valid_mask`
- transmitter and receiver coordinates in meters

## Default Settings

- iterations: `30`
- damping: `0.0`
- reference sound speed: `1500 m/s`
- sound-speed bounds: `[1300, 1700] m/s`

## Expected Failure Modes

- operator scaling errors from wrong grid spacing or coordinate units;
- noisy or unwrapped phase-delay features;
- overfitting sparse or low-SNR rays when damping is zero;
- wrong sign convention producing faster speed for positive delay.

## What To Adjust First

1. Confirm the projector adjoint test passes.
2. Check the sign convention with a slower synthetic inclusion.
3. Add damping if residuals fit noise.
4. Tighten sound-speed bounds when artifacts dominate.

## Acceptance Tests

- homogeneous phantom returns near-water sound speed;
- positive delay through a slower object reconstructs slower speed;
- projector dot-product adjoint test passes.

