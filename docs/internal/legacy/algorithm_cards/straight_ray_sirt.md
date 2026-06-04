# Straight-Ray SIRT

## Physical Assumption

Straight-ray SIRT uses the same travel-time linearization as the other ray sound-speed baselines:

```text
delta_tof_s = integral(1 / c(x) - 1 / c_ref) ds
```

It updates all rays simultaneously with row and column path-length normalization.

## Input Requirements

- `USCTCase.measurement.domain = features`
- `measurement.delta_tof_s` with shape `[n_tx, n_rx]`
- optional `measurement.valid_mask`
- transmitter and receiver coordinates in meters

## Default Settings

- iterations: `50`
- relaxation: `0.3`
- optional post-update smoothing: disabled by default
- optional ROI-only update: disabled by default
- reference sound speed: `1500 m/s`
- sound-speed bounds: `[1300, 1700] m/s`

## Expected Failure Modes

- incorrect travel-time sign convention;
- geometry in millimeters instead of meters;
- valid mask not matching `[n_tx, n_rx]`;
- streaking from sparse angular coverage.

## What To Adjust First

1. Check geometry units and receiver ordering.
2. Inspect the travel-time sinogram and valid mask.
3. Lower relaxation if residuals oscillate.
4. Increase iterations gradually.
5. Enable light smoothing or ROI-only updates only after the sinogram and coverage diagnostics look sane.

## Acceptance Tests

- homogeneous phantom returns the reference sound speed;
- slower inclusion produces a slower reconstructed center;
- data residual remains finite;
- benchmark runs write coverage diagnostics and a residual curve.

## References and Related Code

- Implementation: `src/usctbench/algorithms/ray/sirt.py`
- Projector: `src/usctbench/algorithms/ray/straight_projector.py`
- Tests: `tests/test_projector_adjoint.py`, `tests/test_straight_ray_synthetic.py`
- Background: simultaneous iterative reconstruction for line-integral tomography; see `docs/references.bib` and `docs/algorithm_taxonomy.md`.
