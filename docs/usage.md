# Usage

## Install

~~~bash
pip install -e ".[dev,viz]"
usct list-algorithms --json
~~~

NumPy/SciPy/HDF5/Pydantic/YAML/scikit-image are core dependencies. Plotting is
optional. Numba acceleration is available through the performance extra.
Choose exactly one matching CuPy extra (cuda11 or cuda12) when using Full-Green
on CUDA. WUST has its own MATLAB/MEX dependencies.

## Workspace

Keep source, data, runs and external runtimes separate, for example:

~~~text
workspace/
  usct-benchlab/
  WaveformInversionUST/
  data/
  runs/
  research_archive/
~~~

No Python API depends on a personal workspace path. CLI deployment flags may use
WUST_RUNTIME_ROOT, WUST_MATLAB_EXECUTABLE, WUST_BACKEND and CUDA_VISIBLE_DEVICES.
Suite YAML supports explicit environment substitutions.

## Prepare a deterministic synthetic case

~~~bash
usct data prepare --dataset synthetic --out data/synthetic_demo \
  --shape 32 32 --n-transducers 32 --reference-sound-speed-mps 1500 \
  --inclusion-mps 1450 --inclusion-radius-m 0.006
usct inspect data/synthetic_demo/cases/synthetic_circular_sos.h5
~~~

No RNG is used here. Random holdout selection, if requested, uses evaluation.seed.

## OpenBreastUS property maps

First inspect stored arrays, then specify the actual volume, axes, units and
desired output spacing. The following spacing/axes are examples, not inferred
dataset metadata:

~~~bash
usct inspect /path/to/OpenBreastUS --dataset openbreastus
usct data prepare --dataset openbreastus --root /path/to/OpenBreastUS \
  --files breast_test_speed.mat --sample-indices 0 200 500 700 \
  --sample-axis 2 --image-axes yx --sound-speed-units m/s \
  --spacing-m 0.0003 0.0003 --shape 256 256 --n-transducers 64 \
  --out data/openbreastus_demo
~~~

The output manifest records source files and slice indices. No four-class labels
are guessed from index ranges. This path generates straight-ray ToF, not raw
OpenBreastUS pressure.

## NBPslice2D property maps

~~~bash
usct inspect /path/to/NBPslices2D.zip --dataset nbpslice2d
usct data prepare --dataset nbpslice2d --root /path/to/NBPslices2D.zip \
  --cases-per-class 1 --shape 128 128 --n-transducers 64 \
  --out data/nbpslice2d_demo
~~~

The format reader uses explicit stored type/label and format units. Inspect
manifest.json for the selected cases. Python prepare_nbpslice2d also accepts
explicit case_ids. This is likewise a property-map surrogate.

## Import pressure and extract arrivals

For the supported k-Wave MAT layout, declare stored channel axes:

~~~bash
usct data import-pressure --input /path/to/acquisition.mat \
  --out data/pressure_case.h5 --channel-axes tx,rx,time \
  --reference-sound-speed-mps 1500 --frequencies-hz 150000 250000 350000
~~~

Coordinates/time must exist in the acquisition, or supply an explicit GridSpec
YAML through --grid. Optional --water-dataset names independent water traces.
The importer retains RF, absolute time, source data and selected complex
frequencies; it does not generate observations from the GT map.

Arrival features are a separate explicit preparation step:

~~~python
from usctbench.core.io import read_case_hdf5, write_case_hdf5
from usctbench.core.schema import MeasurementSpec
from usctbench.data.arrival import water_relative_delays, arrival_observation_metadata
import numpy as np

case = read_case_hdf5("data/pressure_case.h5")
m = case.measurement
assert m.water_reference_time is not None
distances = np.linalg.norm(
    case.geometry.tx_pos_m[:, None] - case.geometry.rx_pos_m[None, :], axis=-1
)
# These window settings must match the recorded source pulse, not GT.
delay, valid, weights, qc = water_relative_delays(
    m.time_data.transpose(2, 0, 1), m.water_reference_time.transpose(2, 0, 1),
    m.time_axis_s, distances, reference_speed_mps=1500,
    picker="xcorr", source_onset_s=0.0, pulse_duration_s=15e-6,
)
features = case.model_copy(update={
    "measurement": MeasurementSpec(domain="features", delta_tof_s=delay,
                                   valid_mask=valid, ray_weights=weights),
    "metadata": {**case.metadata, **arrival_observation_metadata(qc)},
})
write_case_hdf5(features, "data/tof_case.h5")
~~~

See [datasets](datasets.md) for the exact callable and required arrays; do not
substitute a speed-map surrogate for missing water/RF observations.

## Run algorithms

Each command below uses an already prepared case and an exact output directory.
Choose the file whose observation matches the algorithm:

~~~bash
usct run straight_cgls --case data/tof_case.h5 --config configs/algorithms/cgls.yaml --out runs/cgls
usct run straight_sirt --case data/tof_case.h5 --config configs/algorithms/sirt.yaml --out runs/sirt
usct run straight_sart --case data/tof_case.h5 --config configs/algorithms/sart.yaml --out runs/sart
usct run bent_ray_gn --case data/first_arrival_case.h5 --config configs/algorithms/bent_ray.yaml --out runs/bent
usct run rwave_adapter --case data/pressure_case.h5 --config configs/algorithms/rwave.yaml --out runs/born
usct run fwi_wust --case data/pressure_case.h5 --config configs/algorithms/fwi_wust.yaml \
  --runtime-root /path/to/WaveformInversionUST --matlab-executable /path/to/matlab \
  --backend cpu --out runs/fwi
~~~

The FWI example needs a grid admitting the configured PML, declared pressure
conventions and bounds appropriate to its supplied initialization.
[FWI deployment](fwi.md) explains those requirements. CPU/GPU is explicit.

## Benchmarks and saved results

~~~bash
usct bench --suite configs/benchmarks/synthetic_demo.yaml
usct bench --suite configs/benchmarks/openbreastus_demo.yaml
usct bench --suite configs/benchmarks/nbpslice2d_demo.yaml
usct evaluate --run runs/cgls
~~~

All suite paths, including glob, configs and output root, are relative to the
suite file, not the shell directory. No matched cases is an error.

~~~python
from usctbench.core.io import read_result_hdf5
from usctbench.viz import write_preview_png

result = read_result_hdf5("runs/cgls/result.h5")
print(result.sound_speed_mps.shape)
write_preview_png(result.sound_speed_mps, "runs/cgls/manual_preview.png")
~~~

Use run.json for final execution/reporting states and metrics.json for evaluation.
The HDF5 arrays are saved before optional scoring/plotting.

## Troubleshooting

- Unknown algorithm/parameter: use list-algorithms and describe-algorithm; removed
  aliases are not silently ignored.
- Missing data or empty glob: inspect the preparation manifest and suite-relative paths.
- Missing FWI runtime: pass runtime root/MATLAB/backend explicitly; describe is
  static, while probe is an explicit WUST diagnostic.
- NaN/Inf: inspect failure stage/message and the actual valid mask. Do not replace
  invalid samples with valid zeros.
- Missing preview: install the viz extra; numerical results remain available.
- Null SSIM/PSNR or absent GT: inspect metric status, not just the scalar field.
