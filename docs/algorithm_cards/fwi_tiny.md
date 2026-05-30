# Tiny FWI

## Physical Assumption

`fwi_tiny` is a synthetic waveform-inversion proof of life. It uses a one-dimensional path travel-time waveform with low-frequency phase observations:

```text
waveform = [cos(2*pi*f*tau), sin(2*pi*f*tau)]
tau = sum(dx / c_i)
```

It is not production FWI and does not model diffraction, PMLs, source wavelets, or full acoustic propagation.

## Input Requirements

- Synthetic `USCTCase`
- `ground_truth.sound_speed_mps`
- grid spacing in meters

## Default Settings

- frequencies: `100, 150, 200 kHz`
- initial sound speed: `1500 m/s`
- steps: `20`
- learning rate: `1e6`
- sound-speed bounds: `[1300, 1700] m/s`

## Expected Failure Modes

- Running on real measured data without waveform observations.
- Phase wrapping when frequencies are too high.
- Non-identifiability along one path; this test only proves gradient plumbing and loss descent.

## What To Adjust First

1. Lower frequencies.
2. Reduce learning rate.
3. Smooth or simplify the initial model.
4. Check gradient sign with `check_tiny_fwi_gradient`.

## Acceptance Tests

- analytic gradient agrees with finite-difference directional derivative;
- gradient descent lowers waveform loss on a synthetic case;
- CLI run writes the standard result artifacts.

## References and Related Code

- Implementation: `src/usctbench/algorithms/fwi/tiny_fwi.py`
- Gradient helper: `src/usctbench/algorithms/fwi/gradient_check.py`
- Tests: `tests/test_fwi_gradient_check.py`, `tests/test_fwi_loss_decrease.py`
- Scope note: this is a tiny synthetic proof-of-life only; see `docs/algorithm_taxonomy.md` for the v0.1 classical-method boundary.
