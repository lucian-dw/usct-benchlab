# usct-benchlab

`usct-benchlab` is a lightweight Python benchmark package for ultrasound
computed tomography (USCT) reconstruction algorithms with unified input/output,
classical baselines, and FWI adapter support. It provides dataset conversion
helpers, runnable algorithm configs, common metrics, preview figures, and
benchmark summaries for reproducible comparisons.

## What is USCT?

**Ultrasound computed tomography is a PDE-constrained inverse problem.** A
source transducer emits an acoustic pulse, the pressure field propagates through
the object according to an acoustic wave equation, and receiver transducers
measure the resulting pressure traces. The inverse problem is to recover
spatial acoustic properties from those measurements.

The main v0.1 target is the sound-speed map $c(x)$. Related physical
properties include density $\rho(x)$ and attenuation $\alpha(x)$. This package
converts datasets to a common `USCTCase` schema and returns every algorithm
output as a `ReconstructionResult`.

## Mathematical Formulation

For source $s$, a simple lossless acoustic pressure model is

$$
\frac{1}{c(x)^2}\partial_{tt}p_s(t,x)-\Delta p_s(t,x)=q_s(t,x).
$$

A more general acoustic model can include density and attenuation:

$$
\begin{aligned}
\frac{1}{c(x)^2}\partial_{tt}p_s
-\nabla\cdot\left(\frac{1}{\rho(x)}\nabla p_s\right)
+\mathcal A_\alpha[p_s]
&= q_s.
\end{aligned}
$$

Receiver $r$ observes the pressure through a measurement operator:

$$
d_{sr}(t)=\mathcal M_r p_s(t,\cdot)+\eta_{sr}(t).
$$

The USCT inverse problem is therefore

$$
\text{recover } c(x),\rho(x),\alpha(x)
\quad
\text{from}
\quad
\{d_{sr}(t)\}_{s,r}.
$$

In v0.1, the primary reconstructed field is $c(x)$.

The straight-ray travel-time approximation uses a reference speed $c_0$ and
ray path $\gamma_{sr}$:

$$
\Delta t_{sr}
\approx
\int_{\gamma_{sr}}
\left(\frac{1}{c(x)}-\frac{1}{c_0}\right)d\ell.
$$

After discretization,

$$
A\delta s \approx b,
\qquad
\delta s = \frac{1}{c}-\frac{1}{c_0}.
$$

CGLS, SIRT, and SART solve algebraic ray systems of the form

$$
\min_{\delta s}
\|W(A\delta s-b)\|_2^2+\lambda^2R(\delta s).
$$

For bent-ray travel time, the eikonal model is

$$
|\nabla T_s(x)| = \frac{1}{c(x)},
\qquad
t_{sr}\approx T_s(r).
$$

The v0.1 `bent_ray_gn` command is a regularized bent-ray-style travel-time
baseline, not a full external eikonal solver.

A schematic weak-scattering ray-Born model is

$$
\delta \hat p_{sr}(\omega)
\approx
\int_\Omega
G_0(\omega,r,x)K_\omega(x)G_0(\omega,x,s)\delta m(x)\,dx.
$$

The v0.1 `rwave_adapter` command is a ray-Born-inspired adapter baseline. It
does not claim full reproduction of an external complex rWave solver.

FWI uses pressure data directly:

$$
\min_c
\frac{1}{2}\sum_{\omega,s,r}
\left|
\hat p_s(\omega,r;c)-\hat p_{sr}^{\mathrm{obs}}(\omega)
\right|^2
+\lambda R(c).
$$

The v0.1 FWI path is an adapter for high-fidelity external k-Wave/FWI results.
For more detail, see [docs/math_formulation.md](docs/math_formulation.md).

## Supported Algorithms

| Algorithm | Command name | Mathematical model | Input requirement | Typical use | Config |
| --- | --- | --- | --- | --- | --- |
| CGLS | `straight_cgls` | Straight-ray weighted least squares | `USCTCase` with ring geometry and travel-time measurements | Fast sound-speed baseline | `configs/algorithms/cgls.yaml` |
| SIRT | `straight_sirt` | Simultaneous iterative ray tomography | `USCTCase` with ring geometry and travel-time measurements | Robust iterative sound-speed baseline | `configs/algorithms/sirt.yaml` |
| SART | `straight_sart` | Ordered/subset algebraic ray update | `USCTCase` with ring geometry and travel-time measurements | Ordered-update straight-ray baseline | `configs/algorithms/sart.yaml` |
| Attenuation SIRT | `attenuation_sirt` | Straight-ray log-amplitude tomography | `USCTCase` with log-amplitude measurements | Attenuation baseline | `configs/algorithms/attenuation.yaml` |
| Bent-ray | `bent_ray_gn` | Regularized bent-ray-style travel-time baseline | `USCTCase` with travel-time measurements | Refraction-style comparison | `configs/algorithms/bent_ray.yaml` |
| rWave adapter | `rwave_adapter` | Ray-Born-inspired adapter baseline | `USCTCase` with travel-time measurements | Wave-inspired adapter comparison | `configs/algorithms/rwave.yaml` |
| FWI adapter | `fwi_kwave_adapter` | PDE-level full-wave inversion adapter | `USCTCase` plus external k-Wave/FWI artifact or command path | High-fidelity FWI reporting | `configs/algorithms/fwi_kwave.yaml` |
| Tiny FWI sanity | `fwi_tiny` | Small waveform-inversion sanity model | Small synthetic sound-speed case | Local waveform-inversion plumbing test | `configs/algorithms/fwi_tiny.yaml` |

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

See [docs/usage.md](docs/usage.md) and [docs/datasets.md](docs/datasets.md)
for more complete workflows.

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

For the FWI adapter, set `USCT_KWAVE_FWI_RESULT_PATH` when the config should
ingest an existing reconstruction artifact.

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
[docs/references.bib](docs/references.bib).
