# usct-benchlab

`usct-benchlab` is a lightweight Python benchmark package for ultrasound computed tomography (USCT) reconstruction algorithms with unified input/output, classical baselines, and FWI adapter support. It provides dataset conversion helpers, runnable algorithm configs, common metrics, preview figures, and benchmark summaries for reproducible comparisons.

## Supported Algorithms

| Algorithm | Command name | Input requirement | Typical use | Config file |
| --- | --- | --- | --- | --- |
| CGLS | `straight_cgls` | `USCTCase` with ring geometry and travel-time measurements | Fast straight-ray sound-speed baseline | `configs/algorithms/cgls.yaml` |
| SIRT | `straight_sirt` | `USCTCase` with ring geometry and travel-time measurements | Robust iterative sound-speed baseline | `configs/algorithms/sirt.yaml` |
| SART | `straight_sart` | `USCTCase` with ring geometry and travel-time measurements | Ordered-update straight-ray baseline | `configs/algorithms/sart.yaml` |
| Bent-ray | `bent_ray_gn` | `USCTCase` with travel-time measurements | Regularized bent-ray-style comparison | `configs/algorithms/bent_ray.yaml` |
| rWave adapter | `rwave_adapter` | `USCTCase` with travel-time measurements | Adapter-style wave-inspired baseline | `configs/algorithms/rwave.yaml` |
| FWI adapter | `fwi_kwave_adapter` | `USCTCase` plus external k-Wave/FWI artifact or command path | High-fidelity FWI result ingestion and reporting | `configs/algorithms/fwi_kwave.yaml` |
| Tiny FWI sanity | `fwi_tiny` | Small synthetic sound-speed case | Local proof-of-life for waveform inversion plumbing | `configs/algorithms/fwi_tiny.yaml` |

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

## Environment

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

Synthetic demo:

```bash
usct data make-synthetic-smoke \
  --out "$USCT_WORKSPACE/data/synthetic_demo" \
  --shape 48 \
  --n-transducers 48
```

## Run One Algorithm

CGLS:

```bash
usct run straight_cgls \
  --case "$USCT_WORKSPACE/data/synthetic_demo/cases/synthetic_circular_sos.h5" \
  --config configs/algorithms/cgls.yaml \
  --out "$USCT_RUN_ROOT/single_cgls"
```

SIRT:

```bash
usct run straight_sirt \
  --case "$USCT_WORKSPACE/data/synthetic_demo/cases/synthetic_circular_sos.h5" \
  --config configs/algorithms/sirt.yaml \
  --out "$USCT_RUN_ROOT/single_sirt"
```

Bent-ray:

```bash
usct run bent_ray_gn \
  --case "$USCT_WORKSPACE/data/synthetic_demo/cases/synthetic_circular_sos.h5" \
  --config configs/algorithms/bent_ray.yaml \
  --out "$USCT_RUN_ROOT/single_bent_ray"
```

rWave adapter:

```bash
usct run rwave_adapter \
  --case "$USCT_WORKSPACE/data/synthetic_demo/cases/synthetic_circular_sos.h5" \
  --config configs/algorithms/rwave.yaml \
  --out "$USCT_RUN_ROOT/single_rwave"
```

FWI adapter:

```bash
usct run fwi_kwave_adapter \
  --case "$USCT_WORKSPACE/data/openbreastus_demo/cases/example_case.h5" \
  --config configs/algorithms/fwi_kwave.yaml \
  --out "$USCT_RUN_ROOT/single_fwi"
```

For the FWI adapter, set `USCT_KWAVE_FWI_RESULT_PATH` when the config should ingest an existing reconstruction artifact.

## Run Benchmarks

```bash
usct bench --suite configs/benchmarks/synthetic_demo.yaml
usct bench --suite configs/benchmarks/nbpslice2d_demo.yaml
usct bench --suite configs/benchmarks/openbreastus_demo.yaml
usct bench --suite configs/benchmarks/fwi_kwave_demo.yaml
```

The demo suites use environment-variable defaults. Override the case globs and run roots when needed:

```bash
export USCT_SYNTHETIC_CASE_GLOB="$USCT_WORKSPACE/data/synthetic_demo/cases/*.h5"
export USCT_NBPSLICE2D_CASE_GLOB="$USCT_WORKSPACE/data/nbpslice2d_demo/cases/*.h5"
export USCT_OPENBREASTUS_CASE_GLOB="$USCT_WORKSPACE/data/openbreastus_demo/cases/*.h5"
export USCT_FWI_CASE_GLOB="$USCT_WORKSPACE/data/fwi_demo/cases/*.h5"
export USCT_RUN_ROOT="$USCT_WORKSPACE/runs/usctbench_runs"
```

## Output Files

Single-algorithm runs write:

```text
<out>/<case_id>/result.h5
<out>/<case_id>/metrics.json
<out>/<case_id>/metadata.yaml
<out>/<case_id>/preview.png
```

Benchmark suites write:

```text
<run_root>/<run_id>/<algorithm>/<case_id>/result.h5
<run_root>/<run_id>/<algorithm>/<case_id>/metrics.json
<run_root>/<run_id>/<algorithm>/<case_id>/metadata.yaml
<run_root>/<run_id>/<algorithm>/<case_id>/preview.png
<run_root>/<run_id>/benchmark_summary.csv
<run_root>/<run_id>/benchmark_report.md
```

`metrics.json` contains per-case image and data-consistency metrics when ground truth and forward measurements are available. `metadata.yaml` records the algorithm, config path, case id, runtime, status, and measurement provenance.

## Example Results

OpenBreastUS four-class comparison:

![OpenBreastUS FWI and baseline comparison](docs/assets/openbreastus_readme_fwi_vs_surrogate.png)

NBPslice2D, 2D Acoustic Numerical Breast Phantoms for USCT:

![NBPslice2D FWI and baseline comparison](docs/assets/nbpslice2d_readme_fwi_vs_surrogate.png)

Different algorithms use different measurement assumptions; interpret result panels together with the algorithm cards and case metadata.

## Troubleshooting

- `algorithm not found`: run `usct list-algorithms` and check the command name.
- Missing `.h5` or `.mat` data: confirm the dataset conversion command completed and that the relevant environment variable points to an existing path.
- FWI result path missing: set `USCT_KWAVE_FWI_RESULT_PATH` or edit `configs/algorithms/fwi_kwave.yaml` to point to the artifact you want to ingest.
- NaN/Inf output: inspect `failure_report.md`, check the case units, and lower the iteration count or relaxation in the algorithm config.
- No cases matched by glob: print the expanded `USCT_*_CASE_GLOB` value and verify that converted cases exist under `data/.../cases/`.
- `matplotlib` or `scikit-image` missing: install the visualization extras with `pip install -e ".[viz]"`.

## Development

```bash
python -m compileall src tests
pytest -q
ruff check .
black src tests
bash scripts/run_smoke.sh
```

Release audit:

```bash
bash scripts/audit_release.py
```

## Citations / Datasets

Please cite the datasets and external tools used in your experiments, including OpenBreastUS, NBPslice2D, k-Wave, and WaveformInversionUST when applicable. See `docs/references.bib` and `docs/EXTERNAL_SOURCES_AND_LICENSES.md` for project notes.
