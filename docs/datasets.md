# Datasets and observations

## Three paths

~~~text
property map + declared grid/geometry/c0
  -> build_surrogate_tof_case -> ToF USCTCase -> ray algorithms

recorded or k-Wave pressure + actual time/geometry/units
  -> convert_kwave_pressure_mat -> time/frequency USCTCase
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
RF uses [tx,rx,time], spectrum [frequency,tx,rx], and prepared sources [frequency,tx].
Distances are meters, speeds m/s, times seconds and frequencies Hz. Imported
pressure retains units and transform normalization in metadata.

The positive-sign DTFT is evaluated at the requested frequencies and **actual**
sample times, including nonzero start time:

$$
P(f)=\Delta t\sum_n p(t_n)e^{+2\pi ift_n}.
$$

No nearest FFT-bin substitution is made. WUST owns conversion to its negative-sign
transform, frequency sorting and transducer snapping. Pressure ratios and
scattered fields are not total-pressure WUST inputs.

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

## Pressure and arrival preparation

[waveforms.py](../src/usctbench/data/waveforms.py) preserves selected channel
slabs, measured time samples and geometry while controlling memory use.
[arrival.py](../src/usctbench/data/arrival.py) provides bounded correlation,
envelope and AIC extraction against independent water observations. Each picker
records its QC and valid mask; no picker guarantees Eikonal-model equivalence.

[calibration.py](../src/usctbench/data/calibration.py) normalizes source shapes
and fits complex source scales from independent water using explicit training
masks. A completely held-out frequency may require separately supplied source
calibration. Specimen validation pressure must not fill that gap.

WUST fits its own per-source/per-frequency scale from admitted specimen
receivers; it does not require the Born source spectrum.

See [usage](usage.md) for supported import commands and [evaluation](evaluation.md)
for provenance and data-isolation rules.
