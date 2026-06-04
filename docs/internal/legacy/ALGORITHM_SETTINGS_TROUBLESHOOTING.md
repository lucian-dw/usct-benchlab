# Algorithm Settings Troubleshooting

This is the root troubleshooting entry point. The longer planning note remains under `docs/project/ALGORITHM_SETTINGS_TROUBLESHOOTING.md`.

## Ray Sound-Speed Methods

1. Check units: positions and spacing in meters, sound speed in `m/s`, delays in seconds.
2. Check sign convention: positive delay through a slower object should reconstruct lower speed.
3. Inspect the travel-time sinogram and valid mask before tuning.
4. Inspect coverage maps, row/column norm ranges, and residual curves before tuning.
5. Lower relaxation for SART/SIRT if residuals oscillate.
6. For CGLS, sweep `regularization_lambda`; use `regularization: laplacian` when rough ring/streak artifacts remain after the data residual is already small.
7. Speed-map surrogate OpenBreastUS/NBPslice2D cases are not measured RF benchmarks. Treat them as projector/metric smoke tests and keep that distinction in reports.

## Attenuation Tomography

1. Confirm `log_amp = log(case/reference)`.
2. Remember the model uses `log_amp = -integral(alpha ds)`.
3. Reject low-SNR or invalid receivers through `valid_mask`.
4. Lower relaxation if nonnegative SIRT clips to zero or diverges.

## MATLAB Adapters

1. Confirm `MATLAB_BIN` or `parameters.matlab_bin`.
2. Confirm `parameters.external_root` and `parameters.entrypoint`.
3. Keep external source outside Git unless license review is complete.
4. Treat `skipped` as expected when MATLAB/external packages are unavailable.

## Tiny FWI

1. Run gradient check before changing the update rule.
2. Lower frequencies if phase wraps.
3. Reduce learning rate if loss increases.
4. Do not treat `fwi_tiny` as production FWI; it is a proof-of-life test.
