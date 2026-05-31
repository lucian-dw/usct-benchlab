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
- regularization: `identity`
- regularization lambda: `0.0` (`damping = lambda^2` for the augmented least-squares system)
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
3. Sweep `regularization_lambda` before increasing iterations.
4. Try `regularization: laplacian` when ring/streak artifacts dominate and data residuals are already small.
5. Tighten sound-speed bounds when artifacts dominate.

## Acceptance Tests

- homogeneous phantom returns near-water sound speed;
- positive delay through a slower object reconstructs slower speed;
- projector dot-product adjoint test passes;
- identity and Laplacian regularization paths run on synthetic cases.

## References and Related Code

- Implementation: `src/usctbench/algorithms/ray/cgls.py`
- Projector: `src/usctbench/algorithms/ray/straight_projector.py`
- Tests: `tests/test_projector_adjoint.py`, `tests/test_straight_ray_synthetic.py`
- Background: conjugate-gradient least squares for linearized line-integral tomography; see `docs/references.bib` and `docs/algorithm_taxonomy.md`.
