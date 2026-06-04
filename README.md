# usct-benchlab

`usct-benchlab` is a lightweight Python benchmark package for ultrasound
computed tomography (USCT) reconstruction algorithms with unified input/output,
classical baselines, and FWI adapter support. It provides dataset conversion
helpers, runnable algorithm configs, common metrics, preview figures, and
benchmark summaries for reproducible comparisons.

## What is USCT?

Ultrasound computed tomography images tissue properties from waves transmitted
through and scattered by the breast. A ring or partial-ring array emits
ultrasound from one transducer and records pressure signals at other receivers.
Reconstruction algorithms estimate acoustic property maps such as sound speed
`c(x)`, density `rho(x)`, and attenuation `alpha(x)`.

In this package, every dataset is converted into a `USCTCase`, and every
algorithm returns a `ReconstructionResult`. This keeps straight-ray methods,
algebraic solvers, adapter-style baselines, and FWI outputs comparable through
one artifact layout.

## Mathematical Formulation

The general USCT inverse problem is written as

```math
d = \mathcal{F}(c,\rho,\alpha;\theta) + \eta,
```

where `d` is receiver-array data or derived features, `theta` contains
acquisition settings, and the primary v0.1 reconstruction target is the
sound-speed map `c(x)`.

Straight-ray travel-time methods use

```math
\Delta t_{ij}
=
\int_{\gamma_{ij}}
\left(\frac{1}{c(x)}-\frac{1}{c_0}\right)\,ds,
```

which discretizes to

```math
A\delta s \approx b,
\qquad
\delta s = 1/c - 1/c_0.
```

CGLS, SIRT, and SART solve regularized algebraic systems of the form

```math
\min_{\delta s}
\|W(A\delta s-b)\|_2^2 + \lambda^2 R(\delta s).
```

CGLS is a Krylov least-squares solver. SIRT and SART are algebraic iterative solvers with different residual/update schedules.

Attenuation tomography uses the log-amplitude approximation

```math
-\log |p/p_0| \approx \int_\gamma \alpha(x)\,ds.
```

Refraction-corrected travel-time methods replace fixed straight paths with a travel-time field:

```math
|\nabla T_i(x)| = 1/c(x),
\qquad
t_{ij} \approx T_i(r_j).
```

The v0.1 `bent_ray_gn` command is a regularized bent-ray-style travel-time baseline. It is not a full external eikonal solver.

The weak-scattering ray-Born model can be written as

```math
\delta p(\omega,r_i,r_j)
\approx
\int_\Omega
G_0(\omega,r_j,x)K_\omega(x)G_0(\omega,x,r_i)\delta m(x)\,dx.
```

Full ray-Born reconstruction requires complex frequency-domain pressure data. The v0.1 `rwave_adapter` command is an adapter-style baseline and does not claim full reproduction of an external rWave MATLAB package.

FWI minimizes waveform or frequency-domain pressure mismatch:

```math
\min_c
\frac{1}{2}\sum_{\omega,i,j}
\|P_{\omega,i,j}^{obs}-P_{\omega,i,j}(c)\|_2^2 + \lambda R(c).
```

The v0.1 FWI path is an adapter for high-fidelity external k-Wave/FWI results. For more details, see [docs/math_formulation.md](docs/math_formulation.md).

## Supported Algorithms

| Algorithm | Command name | Input requirement | Typical use | Config |
| --- | --- | --- | --- | --- |
| CGLS | `straight_cgls` | `USCTCase` with ring geometry and travel-time measurements | Fast straight-ray sound-speed baseline | `configs/algorithms/cgls.yaml` |
| SIRT | `straight_sirt` | `USCTCase` with ring geometry and travel-time measurements | Robust iterative sound-speed baseline | `configs/algorithms/sirt.yaml` |
| SART | `straight_sart` | `USCTCase` with ring geometry and travel-time measurements | Ordered-update straight-ray baseline | `configs/algorithms/sart.yaml` |
| Attenuation SIRT | `attenuation_sirt` | `USCTCase` with log-amplitude measurements | Straight-ray attenuation baseline | `configs/algorithms/attenuation.yaml` |
| Bent-ray | `bent_ray_gn` | `USCTCase` with travel-time measurements | Regularized bent-ray-style comparison | `configs/algorithms/bent_ray.yaml` |
| rWave adapter | `rwave_adapter` | `USCTCase` with travel-time measurements | rWave/ray-Born-inspired adapter baseline | `configs/algorithms/rwave.yaml` |
| FWI adapter | `fwi_kwave_adapter` | `USCTCase` plus external k-Wave/FWI artifact or command path | High-fidelity FWI result ingestion and reporting | `configs/algorithms/fwi_kwave.yaml` |
| Tiny FWI sanity | `fwi_tiny` | Small synthetic sound-speed case | Local proof-of-life for waveform inversion plumbing | `configs/algorithms/fwi_tiny.yaml` |

More details are in [docs/algorithms.md](docs/algorithms.md).

## Installation

Conda workflow:

```bash
conda create -n usctbench python=3.10 -y
conda activate usctbench
pip install -e ".[dev,viz]"
```

Pip workflow:

```bash
pip install -r requirements.txt
pip install -e .
```

Check the installation:

```bash
usct --help
usct list-algorithms
pytest -q
```

## Environment and Workspace Layout

Use environment variables so local data and generated runs stay outside Git:

```bash
export USCT_WORKSPACE=/path/to/usct-benchlab
export USCT_DATA_ROOT=$USCT_WORKSPACE/data/openbreastus
export USCT_RUN_ROOT=$USCT_WORKSPACE/runs/usctbench_runs
export USCT_NBP_ZIP_PATH=/path/to/NBPslices2D.zip
export USCT_KWAVE_FWI_RESULT_PATH=/path/to/fwi_result.mat
```

Recommended workspace layout:

```text
<workspace>/
  code/          # this repository
  data/          # local datasets and converted cases
  runs/          # benchmark outputs
  external/      # optional external projects
  checkpoints/   # local weights or checkpoints
```

## Prepare Datasets

Synthetic demo:

```bash
usct data make-synthetic-smoke \
  --out "$USCT_WORKSPACE/data/synthetic_demo" \
  --shape 48 \
  --n-transducers 48
```

OpenBreastUS:

```bash
usct data inspect-openbreastus \
  --root "$USCT_DATA_ROOT" \
  --out "$USCT_RUN_ROOT/openbreastus_index.json"

usct data make-quality \
  --root "$USCT_DATA_ROOT" \
  --out "$USCT_WORKSPACE/data/openbreastus_demo" \
  --cases-per-density 1 \
  --converted-shape 256 \
  --n-transducers 128
```

NBPslice2D:

```bash
usct data inspect-nbpslice2d \
  --zip "$USCT_NBP_ZIP_PATH" \
  --out "$USCT_RUN_ROOT/nbpslice2d_index.json"

usct data make-nbp-quality \
  --zip "$USCT_NBP_ZIP_PATH" \
  --out "$USCT_WORKSPACE/data/nbpslice2d_demo" \
  --cases-per-type 1 \
  --converted-shape 256 \
  --n-transducers 128
```

See [docs/usage.md](docs/usage.md) and [docs/datasets.md](docs/datasets.md) for more complete workflows.

## Run One Algorithm

CGLS:

```bash
usct run straight_cgls \
  --case "$USCT_WORKSPACE/data/synthetic_demo/cases/synthetic_circular_sos.h5" \
  --config configs/algorithms/cgls.yaml \
  --out runs/single_cgls
```

SIRT:

```bash
usct run straight_sirt \
  --case "$USCT_WORKSPACE/data/synthetic_demo/cases/synthetic_circular_sos.h5" \
  --config configs/algorithms/sirt.yaml \
  --out runs/single_sirt
```

SART:

```bash
usct run straight_sart \
  --case "$USCT_WORKSPACE/data/synthetic_demo/cases/synthetic_circular_sos.h5" \
  --config configs/algorithms/sart.yaml \
  --out runs/single_sart
```

Bent-ray:

```bash
usct run bent_ray_gn \
  --case "$USCT_WORKSPACE/data/synthetic_demo/cases/synthetic_circular_sos.h5" \
  --config configs/algorithms/bent_ray.yaml \
  --out runs/single_bent_ray
```

rWave adapter:

```bash
usct run rwave_adapter \
  --case "$USCT_WORKSPACE/data/synthetic_demo/cases/synthetic_circular_sos.h5" \
  --config configs/algorithms/rwave.yaml \
  --out runs/single_rwave
```

FWI adapter:

```bash
usct run fwi_kwave_adapter \
  --case "$USCT_WORKSPACE/data/openbreastus_demo/cases/example_case.h5" \
  --config configs/algorithms/fwi_kwave.yaml \
  --out runs/single_fwi
```

For the FWI adapter, set `USCT_KWAVE_FWI_RESULT_PATH` when the config should ingest an existing reconstruction artifact.

## Run Benchmarks

```bash
usct bench --suite configs/benchmarks/synthetic_demo.yaml
usct bench --suite configs/benchmarks/nbpslice2d_demo.yaml
usct bench --suite configs/benchmarks/openbreastus_demo.yaml
usct bench --suite configs/benchmarks/fwi_kwave_demo.yaml
```

Demo suites read these optional case globs:

```bash
export USCT_SYNTHETIC_CASE_GLOB="$USCT_WORKSPACE/data/synthetic_demo/cases/*.h5"
export USCT_NBP_CASE_GLOB="$USCT_WORKSPACE/data/nbpslice2d_demo/cases/*.h5"
export USCT_OPENBREASTUS_CASE_GLOB="$USCT_WORKSPACE/data/openbreastus_demo/cases/*.h5"
export USCT_KWAVE_FWI_CASE_GLOB="$USCT_WORKSPACE/data/fwi_kwave_demo/cases/*.h5"
```

## Output Files

Single-algorithm runs write:

```text
runs/single_cgls/synthetic_circular_sos/result.h5
runs/single_cgls/synthetic_circular_sos/metrics.json
runs/single_cgls/synthetic_circular_sos/metadata.yaml
runs/single_cgls/synthetic_circular_sos/preview.png
```

Benchmark suites write:

```text
runs/usctbench_runs/synthetic_demo_YYYYMMDDTHHMMSSZ/straight_cgls/synthetic_circular_sos/result.h5
runs/usctbench_runs/synthetic_demo_YYYYMMDDTHHMMSSZ/straight_cgls/synthetic_circular_sos/metrics.json
runs/usctbench_runs/synthetic_demo_YYYYMMDDTHHMMSSZ/straight_cgls/synthetic_circular_sos/metadata.yaml
runs/usctbench_runs/synthetic_demo_YYYYMMDDTHHMMSSZ/straight_cgls/synthetic_circular_sos/preview.png
runs/usctbench_runs/synthetic_demo_YYYYMMDDTHHMMSSZ/benchmark_summary.csv
runs/usctbench_runs/synthetic_demo_YYYYMMDDTHHMMSSZ/benchmark_report.md
```

`metrics.json` contains per-case image and data-consistency metrics when ground
truth and forward measurements are available. `metadata.yaml` records the
algorithm, config path, case id, runtime, status, and measurement provenance.

## Example Results

OpenBreastUS four-class comparison:

![OpenBreastUS FWI and baseline comparison](docs/assets/openbreastus_readme_fwi_vs_surrogate.png)

NBPslice2D, 2D Acoustic Numerical Breast Phantoms for USCT:

![NBPslice2D FWI and baseline comparison](docs/assets/nbpslice2d_readme_fwi_vs_surrogate.png)

Different algorithms use different measurement assumptions; interpret result
panels together with [docs/algorithms.md](docs/algorithms.md) and case
metadata.

## Troubleshooting

- `algorithm not found`: run `usct list-algorithms` and check the command name.
- Missing `.h5` or `.mat` data: confirm the dataset conversion command
  completed and that the relevant environment variable points to an existing
  path.
- FWI result path missing: set `USCT_KWAVE_FWI_RESULT_PATH` or edit
  `configs/algorithms/fwi_kwave.yaml` to point to the artifact you want to
  ingest.
- NaN/Inf output: inspect `failure_report.md`, check the case units, and lower
  the iteration count or relaxation in the algorithm config.
- No cases matched by glob: print the expanded `USCT_*_CASE_GLOB` value and
  verify that converted cases exist under `data/.../cases/`.
- `matplotlib` or `scikit-image` missing: install the visualization extras with
  `pip install -e ".[viz]"`.

## Development

```bash
black src tests
ruff check src tests --fix
python -m compileall src tests
pytest -q
bash scripts/run_smoke.sh
bash scripts/audit_release.py
```

See [docs/development.md](docs/development.md) for release checks and
repository hygiene.

## Citations / Datasets

Please cite the datasets and external tools used in your experiments, including
OpenBreastUS, NBPslice2D, k-Wave, and WaveformInversionUST when applicable. See
[docs/references.bib](docs/references.bib) and
[docs/EXTERNAL_SOURCES_AND_LICENSES.md](docs/EXTERNAL_SOURCES_AND_LICENSES.md).
