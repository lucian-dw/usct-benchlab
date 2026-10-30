# Property-map conversion

Current input paths and conventions are maintained in [datasets.md](datasets.md).
OpenBreastUS selection is explicit: files, dataset_name and sample indices, sample axis, image
axes, units and output spacing. Inspection does not select cases or assign labels.

Continuous maps and nearest-neighbor labels share physical cell-center sampling.
Grid origins are pixel edges. Pressure import requires recorded geometry and
sample times, and never replaces missing observations with a GT-generated ToF.

Property-map conversion calls build_surrogate_tof_case with an explicit medium,
grid, geometry and reference sound speed. It preserves unrelated output files.
