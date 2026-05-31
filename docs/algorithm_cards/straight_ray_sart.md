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
- relaxation: `0.2`
- subsets: `8`
- optional post-update smoothing: disabled by default
- optional ROI-only update: disabled by default
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
5. Enable light smoothing or ROI-only updates only after the sinogram and coverage diagnostics look sane.

## Acceptance Tests

- homogeneous phantom returns the reference sound speed;
- straight-ray projector adjoint dot-product test passes;
- positive delay through a slower object reconstructs a slower center.
- benchmark runs write coverage diagnostics and a residual curve.

## References and Related Code

- Implementation: `src/usctbench/algorithms/ray/sart.py`
- Projector: `src/usctbench/algorithms/ray/straight_projector.py`
- Tests: `tests/test_projector_adjoint.py`, `tests/test_straight_ray_synthetic.py`
- Background: classical algebraic reconstruction methods for tomography; see `docs/references.bib` and `docs/algorithm_taxonomy.md`.
