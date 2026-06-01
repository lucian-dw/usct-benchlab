# Simulation QC

Every k-Wave unified simulation must write these artifacts next to the wavefield case:

- `simulation_qc.json`
- `wavefield_preview.png`
- `source_spectrum.png`
- `receiver_energy_hist.png`
- `water_tof_error.png`
- `reciprocity_error.png`
- `boundary_energy.png`
- `feature_preview.png`

Required QC metrics:

- `grid_points_per_wavelength_min`
- `cfl_number`
- `pml_thickness_pixels`
- `source_peak_frequency_hz`
- `source_bandwidth_hz`
- `receiver_signal_energy_min/median/max`
- `bad_receiver_fraction`
- `reciprocity_error`
- `water_tof_rmse_vs_geometry`
- `water_tof_raw_rmse_vs_geometry`
- `water_tof_bias_s`
- `water_tof_affine_residual_s`
- `water_tof_affine_effective_speed_mps`
- `water_tof_distance_correlation`
- `phase_unwrap_failure_fraction`
- `tof_valid_fraction`
- `amplitude_dynamic_range_db`
- `nan_inf_count`
- `boundary_energy_fraction`

QC metrics are computed only on traces allowed by `measurement.valid_mask`; square
ring cases also exclude diagonal self-pairs. This prevents transmitter/receiver
self-coupling traces from dominating receiver energy, reciprocity, and boundary
energy checks.

For real k-Wave runs, `water_tof_rmse_vs_geometry` removes the median water
peak-time bias before comparing against ring geometry. The raw value is still
reported as `water_tof_raw_rmse_vs_geometry`, and `water_tof_bias_s` records the
global source/group-delay offset that was removed. Differential ToF features use
the water/reference pair, so this global offset should cancel.

If the strict `0.5 * dt` water geometry check fails but the homogeneous-water
arrivals fit an affine time-distance model with correlation at least `0.999` and
residual at most `2 * dt`, QC records a warning instead of a hard failure. This
covers finite grid/source/aperture offsets while still catching broken geometry
or reference handling.

If QC fails, the case metadata is stamped with `simulation_failed_qc: true`. The benchmark runner skips algorithm execution for that case and writes `failure_reason=simulation_failed_qc`, so downstream reports cannot treat the reconstruction as a valid conclusion.

The local `native_smoke` backend validates data plumbing and QC logic. Formal high-fidelity evidence should be generated on A100 with an actual k-Wave backend and the same QC contract.
