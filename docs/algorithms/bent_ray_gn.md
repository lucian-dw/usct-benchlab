# Bent-Ray GN Unified Input

`bent_ray_gn` consumes the same `USCTCase` feature fields as the straight-ray baselines:

- preferred: `delta_tof_s` from cross-correlation ToF or multi-frequency phase-slope delay
- required: `valid_mask`
- optional diagnostics: `tof_first_arrival_s`, `tof_xcorr_s`, `phase_slope_delay_s`, `feature_quality`

Single-frequency phase is not accepted as the default formal ToF. Feature extraction must have water/reference handling and should compare first-arrival, xcorr, and phase-slope delays before benchmark reporting.

When the backend is the project-native solver, it remains a refraction-corrected travel-time surrogate. External MATLAB GN can still be called through the adapter contract.
