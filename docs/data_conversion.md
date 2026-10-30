# Property-map conversion

Current input paths and conventions are maintained in [datasets.md](datasets.md).
OpenBreastUS selection is explicit: files, dataset_name and sample indices, sample axis, image
axes, units and output spacing. Inspection does not select cases or assign labels.

Continuous maps and nearest-neighbor labels share physical cell-center sampling.
The maintained helpers are `resample_bilinear` and `resample_labels_nearest` in
`data/conversion.py`; both preserve the cell-edge field of view.
Grid origins are pixel edges. Pressure import requires recorded geometry and
sample times, and never replaces missing observations with a GT-generated ToF.
`grid_from_coordinates` holds the uniform-coordinate to pixel-edge conversion
used by the pressure reader and its optional ground-truth grid.

Property-map conversion calls build_surrogate_tof_case with an explicit medium,
grid, geometry and reference sound speed. It preserves unrelated output files.
NBPslices2D reads each required array once and records any label-derived ROI as
a reconstruction prior. OpenBreastUS supports both MATLAB v5 and v7.3/HDF5 files.

Pressure reading and saving are separate:

~~~python
from usctbench.core.io import write_case_hdf5
from usctbench.data.waveforms import read_kwave_pressure_mat

case = read_kwave_pressure_mat(
    "pressure.mat", frequencies_hz=[150e3, 200e3], water_dataset="water_dataset"
)
# Attach any derived features and acquisition QC before the single save.
write_case_hdf5(case, "pressure_case.h5")
~~~

Time-only reading does not compute a spectrum or populate frequency water data.
Simulation QC has one source, `metadata.simulation_qc.passed`.
