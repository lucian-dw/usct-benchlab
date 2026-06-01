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
- `phase_unwrap_failure_fraction`
- `tof_valid_fraction`
- `amplitude_dynamic_range_db`
- `nan_inf_count`
- `boundary_energy_fraction`

If QC fails, the case metadata is stamped with `simulation_failed_qc: true`. The benchmark runner skips algorithm execution for that case and writes `failure_reason=simulation_failed_qc`, so downstream reports cannot treat the reconstruction as a valid conclusion.

The local `native_smoke` backend validates data plumbing and QC logic. Formal high-fidelity evidence should be generated on A100 with an actual k-Wave backend and the same QC contract.
