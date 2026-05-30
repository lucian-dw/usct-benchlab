# Attenuation Tomography

## Physical Assumption

The attenuation baseline treats log-amplitude ratios as straight-ray line integrals:

```text
log_amp = log(case / reference) = -integral(alpha(x) ds)
```

The reconstructed field is attenuation in `Np/m`.

## Input Requirements

- `USCTCase.measurement.domain = features`
- `measurement.log_amp` with shape `[n_tx, n_rx]` or `[n_freq, n_tx, n_rx]`
- `measurement.valid_mask` when low-SNR or invalid rays should be excluded
- transmitter and receiver positions in meters

## Default Settings

- iterations: `50`
- relaxation: `0.8`
- attenuation upper bound: `80 Np/m`

## Expected Failure Modes

- reference waveform mismatch;
- low-SNR receivers dominating the sinogram;
- amplitude convention flipped relative to the expected `log(case/reference)`;
- attenuation and focusing effects being mixed into one straight-ray model.

## What To Adjust First

1. Inspect log-amplitude ratios before reconstruction.
2. Reject invalid or low-SNR receivers.
3. Lower relaxation.
4. Clip extreme log-amplitude ratios before inversion.

## Acceptance Tests

- synthetic attenuation reconstruction reduces the log-amplitude residual;
- reconstructed attenuation remains non-negative;
- data residual is reported in `metrics.json`.

