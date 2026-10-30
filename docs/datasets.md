# Datasets and observations

## Three paths

~~~text
property map + declared grid/geometry/c0
  -> build_surrogate_tof_case -> ToF USCTCase -> ray algorithms

recorded or k-Wave pressure + actual time/geometry/units
  -> read_kwave_pressure_mat -> time/frequency USCTCase
  -> explicit water-relative arrival extraction -> ray algorithms
  -> calibrated complex pressure -> Born
  -> total complex pressure -> WUST

matched operator fixture
  -> forward(model) -> synthetic observation -> numerical regression
~~~

Surrogate generation receives the explicit medium; reconstruction receives only
observations and a declared initial/reference model. The runner removes the GT
field before numerical execution.

## Conventions

Images, spacing and pixel-edge origins use [y,x]. Positions use [element,yx].
RF uses [time,tx,rx], spectrum [frequency,tx,rx], and prepared sources [frequency,tx].
Generic speed volumes require an explicit `dataset_name`. Pressure imports do
not generate observations from C: labels are attached only with acquisition
coordinates or an explicit `truth_grid`. Missing label coordinates leave a
usable pressure case with a recorded GT attachment reason.
Distances are meters, speeds m/s, times seconds and frequencies Hz. Imported
pressure retains units and transform normalization in metadata.

The positive-sign DTFT is evaluated at the requested frequencies and **actual**
sample times, including nonzero start time:

$$
P(f)=\Delta t\sum_n p(t_n)e^{+2\pi ift_n}.
$$

No nearest FFT-bin substitution is made. `metadata.pressure_contract` is the
single frequency-pressure interpretation: `fourier_sign` is the DTFT kernel sign,
`real_pressure` declares real time-domain pressure, `pressure_type` identifies
total pressure, `data_units` records spectrum units, and `spectrum_normalization`
is `dtft_dt`, `discrete_sum`, or explicitly `unknown`. A `dtft_dt` transform of Pa
samples has Pa*s units. Frequency HDF5 datasets use this contract, not the old
flat unit or harmonic-convention fields. Unknown acquisition units stay unknown.

`prepare_born_pressure(case)` returns a new +DTFT total-pressure case view.
For declared real -DTFT input it conjugates `freq_data`, `source_spectrum`, and
`water_reference` together, preserving units and normalization. Original arrays
and metadata are not mutated; unchanged arrays are shared by convention.
The prepared `measurement.water_reference` is the frequency reference for phase
initialization. Missing calibration arrays remain missing. A complete contract
is required; scattered/ratio pressure and sign conversion of complex time-domain
signals are rejected. WUST owns conversion to its negative-sign transform,
frequency sorting and transducer snapping.

## Property-map readers

OpenBreastUS: inspect independently; select volume files and zero-based slices
explicitly. Declare sample axis, in-plane orientation, speed units and output
spacing. No inferred density groups or silent source links.

NBPslices2D 2025: property maps contain sos/den/att/y/label. The maintained speed
reader uses sos and label/type; unused absorption maps are not reconstructed.
The related measurement collections are separate inputs, not assumed present.

Continuous maps are resampled at physical cell centers; labels use nearest
neighbors. Missing labels remain missing. Missing source coordinates are not
replaced by invented spacing. Unrelated output files are retained.

ROI provenance is explicit in `roi_source` and `roi_use`. NBPslices2D uses its
positive ground-truth labels as a reconstruction prior, not a label-free domain;
missing positive labels leave the reconstruction unrestricted. Synthetic and
OpenBreastUS speed-map cases use the full grid. Pressure cases never derive ROI
from C; an explicitly supplied grid mask is a caller-provided prior. Synthetic
inclusion parameters belong to the preparation manifest's `generation` record,
not measurement metadata.

## Pressure and arrival preparation

[waveforms.py](../src/usctbench/data/waveforms.py) preserves selected channel
slabs, measured time samples and geometry while controlling memory use.
`read_kwave_pressure_mat(mat_path, *, ...)` returns a case without writing files.
The caller attaches derived arrivals/QC and invokes `write_case_hdf5` once.
`water_reference_time` contains independent RF; `water_reference` contains only
its spectrum and remains `None` for time-only imports.
[arrival.py](../src/usctbench/data/arrival.py) provides bounded correlation,
envelope and AIC extraction against independent water observations. Each picker
records its QC and valid mask; no picker guarantees Eikonal-model equivalence.

[calibration.py](../src/usctbench/data/calibration.py) normalizes source shapes
and fits complex source scales from independent water using explicit training
masks. A completely held-out frequency may require separately supplied source
calibration. Specimen validation pressure must not fill that gap.

Phase-slope initialization only fits training frequencies. Direct unwrap
requires `max(diff(f_train)) * max(abs(delay_bounds)) < 0.5` per channel.
The minimum object/reference amplitude ratio is nonnegative; zero admits any
strictly positive object and water amplitudes, but never a zero signal.
Unsupported channels have zero confidence and are counted in
`unwrap_range_rejected_channels`; a small phase-fit RMS alone does not establish
that the selected phase branch is physically identifiable.

WUST fits its own per-source/per-frequency scale from admitted specimen
receivers; it does not require the Born source spectrum.

See [usage](usage.md) for supported import commands and [evaluation](evaluation.md)
for provenance and data-isolation rules.
