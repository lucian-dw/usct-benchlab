# AGENTS.md — `usct-benchlab` Codex instructions

This repository is `usct-benchlab`, a USCT traditional algorithm benchmark library for a medical imaging agent.

v0.1 is **traditional-first**. The priority is to build a reliable benchmark harness and run classical USCT algorithms before any generative model work.

Do **not** implement diffusion, GAN, score-based models, large neural-operator training, or other heavy generative methods in v0.1. Keep extension points for learning-based methods, but the acceptance target is classical USCT algorithms.

---

## 0. Real development workflow

The user works from a local Mac, but the Mac is **not** the authoritative scientific-computing environment.

Use this workflow:

```text
Mac local = lightweight editing / documentation / control / small tests
GitHub    = synchronization bridge
A100      = authoritative runtime environment for CUDA, OpenBreastUS, MATLAB-related wrappers, benchmarks, and heavy numerical debugging
```

Preferred loop:

```text
Mac edit -> git commit/push -> GitHub -> A100 git pull -> A100 run tests/benchmarks
```

If A100 reveals bugs, either:

1. fix locally on Mac, commit/push, then pull on A100; or
2. patch directly on A100, commit/push, then pull back on Mac.

Avoid simultaneous uncommitted edits on both Mac and A100. Before editing on either side, run `git status`. Before running on A100, run `git pull`.

---

## 1. Known paths and repository layout

The local Mac workspace is usually:

```text
/Users/wudalong/Desktop/usct-benchlab
```

The A100 workspace is usually:

```text
~/usct-benchlab
```

The Git repository may be either the workspace root or a `code/` subdirectory. Always detect it using:

```bash
git rev-parse --show-toplevel
```

Preferred split layout:

```text
<workspace>/
  code/          # Git repository, if split layout is used
  data/          # OpenBreastUS and smoke subsets, never committed
  runs/          # benchmark outputs, never committed
  external/      # third-party repos, never committed unless deliberately vendored and licensed
  checkpoints/   # model weights/checkpoints, never committed
```

If the repository root itself is `<workspace>/usct-benchlab`, still use the same logical layout and keep data/runs/checkpoints ignored by Git.

Do not use `/data/...`. The user does not have sudo permission on the A100 server.

Use environment variables and workspace-relative paths:

```bash
export USCT_WORKSPACE=<workspace>
export USCT_DATA_ROOT=$USCT_WORKSPACE/data/openbreastus
export USCT_SAMPLE_ROOT=$USCT_WORKSPACE/data/openbreastus_sample
export USCT_RUN_ROOT=$USCT_WORKSPACE/runs/usctbench_runs
```

On Mac, `USCT_WORKSPACE` should usually be:

```bash
/Users/wudalong/Desktop/usct-benchlab
```

On A100, `USCT_WORKSPACE` should usually be:

```bash
$HOME/usct-benchlab
```

---

## 2. Remote A100 behavior through SSH MCP

Codex is normally controlled locally on Mac. When A100 execution is needed, use the configured SSH MCP connection to run commands remotely.

Before any A100 run, execute remotely:

```bash
cd ~/usct-benchlab/code 2>/dev/null || cd ~/usct-benchlab
git status
git pull --ff-only || git pull
bash scripts/setup_workspace.sh || true
nvidia-smi
python --version
```

A100 is responsible for:

- OpenBreastUS inspection and indexing;
- smoke/mini subset generation;
- CUDA/PyTorch installation checks;
- MATLAB/external adapter checks when available;
- straight-ray smoke benchmarks;
- attenuation benchmarks;
- FWI synthetic tests and later FWI smoke benchmarks;
- long-running numerical debugging.

Mac is responsible for:

- documentation;
- lightweight code editing;
- interface/schema design;
- static checks;
- small synthetic tests that do not require CUDA, MATLAB, or the full OpenBreastUS dataset.

Never require the Mac to install CUDA, MATLAB, k-Wave, Deepwave, or the full OpenBreastUS dataset.

---

## 3. Non-negotiable constraints

1. **No data in Git.** Never commit OpenBreastUS files, checkpoints, generated wavefields, raw `.mat` data, `.h5` cases, `.npy/.npz` arrays, or benchmark runs.
2. **Use the private remote**: `git@github.com:Math-Wu/usct-benchlab.git`.
3. **Every algorithm must use the same I/O**: `USCTCase -> ReconstructionResult`.
4. **Every runnable algorithm must have tests, metrics, configs, and an algorithm card.**
5. **Prefer small verified steps.** Do not build a huge FWI system before the loader, projector, and metrics pass.
6. **Do not silently change units.** Sound speed is `m/s`; slowness is `s/m`; attenuation is `Np/m` unless explicitly converted; frequency is `Hz`; coordinates are meters.
7. **External code must be isolated** under `external/` or wrapped by an adapter. Check license before vendoring code. Prefer submodules or documented install steps.
8. **OpenBreastUS is already downloaded by the user on A100.** Do not download the full dataset. Inspect and index the existing local copy.
9. **If OpenBreastUS structure differs from assumptions**, write a schema-inspection report and adapt the loader. Do not hard-code guessed paths.
10. **If an experiment fails**, write the failure and next actions to `runs/<run_id>/failure_report.md` instead of hiding it.
11. **Do not use sudo** unless the user explicitly says sudo is available. Assume no sudo on A100.
12. **Do not start v0.1 with diffusion/generative models.** They belong in a future roadmap only.

---

## 4. Git discipline

Use short feature branches for nontrivial changes:

```bash
git checkout -b feat/schema-loader
git checkout -b feat/openbreastus-inspector
git checkout -b feat/straight-ray-sart
git checkout -b feat/metrics-benchmark
git checkout -b feat/fwi-tiny
```

Commit only source, configs, tests, docs, and small synthetic fixtures. Do not commit files under:

```text
data/
runs/
checkpoints/
external/           # unless intentionally adding a small licensed adapter, not a full external repo
third_party/
*.mat
*.h5
*.hdf5
*.npy
*.npz
*.zarr/
*.pt
*.pth
*.ckpt
```

If data or run outputs were accidentally staged, unstage them:

```bash
git rm -r --cached data runs checkpoints external third_party 2>/dev/null || true
git status
```

---

## 5. Project architecture target

Build a package with a stable agent-facing CLI:

```bash
usct data inspect-openbreastus --root $USCT_DATA_ROOT
usct data make-smoke --root $USCT_DATA_ROOT --out $USCT_SAMPLE_ROOT --cases-per-density 1
usct list-algorithms
usct run straight_sart --case <case.h5> --config configs/algorithms/straight_sart.yaml --out <run_dir>
usct eval --run <run_dir> --protocol configs/benchmarks/openbreastus_smoke.yaml
usct bench --suite configs/benchmarks/openbreastus_smoke.yaml
```

Every algorithm run must write:

```text
runs/<run_id>/<case_id>/result.h5
runs/<run_id>/<case_id>/metrics.json
runs/<run_id>/<case_id>/metadata.yaml
runs/<run_id>/<case_id>/preview.png
```

Core interfaces:

```python
class USCTCase(BaseModel): ...
class ReconstructionResult(BaseModel): ...
class AlgorithmConfig(BaseModel): ...
class Algorithm(Protocol):
    name: str
    def run(self, case: USCTCase, config: AlgorithmConfig) -> ReconstructionResult: ...
```

---

## 6. Implementation order

Follow this order. Do not start with FWI or deep learning.

1. Repository skeleton, package install, tests, CLI.
2. `USCTCase` and `ReconstructionResult` schema.
3. HDF5 read/write helpers.
4. Algorithm registry.
5. OpenBreastUS inspector and smoke subset creator.
6. Feature extraction for ray methods:
   - phase-delay / travel-time-like feature;
   - log-amplitude ratio;
   - valid mask;
   - reference/water handling when available.
7. Metrics and benchmark runner.
8. Straight-ray projector with adjoint test.
9. Straight-ray SART/SIRT/CGLS sound-speed reconstruction.
10. Straight-ray attenuation tomography.
11. Synthetic fixtures and strict tests.
12. Optional MATLAB adapters for refraction-corrected GN and r-Wave/ray-Born.
13. Tiny FWI with synthetic gradient check and loss-decrease test.
14. Algorithm cards and troubleshooting documentation.

---

## 7. Agent roles

### Agent 0 — Architecture lead

Mission: create the package skeleton and registry used by all other agents.

Deliverables:

- `pyproject.toml`
- `src/usctbench/`
- `src/usctbench/schema.py`
- `src/usctbench/registry.py`
- `src/usctbench/cli.py`
- `src/usctbench/io/hdf5.py`
- `tests/test_schema_roundtrip.py`
- `docs/architecture.md`

### Agent 1 — OpenBreastUS data lead

Mission: turn local OpenBreastUS data into a stable benchmark subset.

Deliverables:

- `src/usctbench/data/openbreastus.py`
- `src/usctbench/data/features.py`
- `src/usctbench/data/smoke_subset.py`
- `data/README.md` if needed, but keep actual data ignored
- `docs/OPENBREASTUS_DATA_PROTOCOL.md`
- `tests/test_openbreastus_inspection.py`

Tasks:

1. Inspect the actual local tree at `$USCT_DATA_ROOT` on A100.
2. Generate `openbreastus_index.json` with case id, density class, speed file path, wavefield paths, available frequencies, shape, and split.
3. Create `openbreastus-smoke-v1`: one small case from each density class if available.
4. Convert selected cases to standard `USCTCase` HDF5 files.
5. Save intermediate diagnostic plots and masks.

### Agent 2 — Metrics and benchmark lead

Mission: define what “runs successfully” means.

Deliverables:

- `src/usctbench/metrics/image.py`
- `src/usctbench/metrics/data_consistency.py`
- `src/usctbench/benchmark/runner.py`
- `src/usctbench/benchmark/report.py`
- `configs/benchmarks/openbreastus_smoke.yaml`
- `configs/benchmarks/openbreastus_mini.yaml`
- `tests/test_metrics.py`

Minimum metrics:

- ROI RMSE, MAE, NRMSE for sound speed.
- SSIM and PSNR where valid.
- Background/water baseline improvement.
- Data residual for algorithms with a forward model.
- Runtime, memory, number of iterations.
- Pass/fail fields with explicit reasons.

### Agent 3 — Straight-ray tomography lead

Mission: implement the reliable baseline first.

Deliverables:

- `src/usctbench/algorithms/ray/straight_projector.py`
- `src/usctbench/algorithms/ray/sart.py`
- `src/usctbench/algorithms/ray/sirt.py`
- `src/usctbench/algorithms/ray/cgls.py`
- `configs/algorithms/straight_sart.yaml`
- `docs/algorithm_cards/straight_ray_sart.md`
- `tests/test_projector_adjoint.py`
- `tests/test_straight_ray_synthetic.py`

Key tests:

- homogeneous phantom returns near-water sound speed;
- dot-product adjoint test passes with relative error `<1e-4`;
- sign convention test: positive delay through slower object should reconstruct slower speed.

### Agent 4 — Attenuation tomography lead

Mission: implement a simple attenuation baseline.

Deliverables:

- `src/usctbench/algorithms/ray/attenuation.py`
- `configs/algorithms/attenuation_sirt.yaml`
- `docs/algorithm_cards/attenuation_tomography.md`
- `tests/test_attenuation_synthetic.py`

Use log-amplitude ratios and robust clipping. Treat attenuation as a separate baseline; do not couple it to FWI in v0.1.

### Agent 5 — Refraction and ray-Born lead

Mission: wrap or minimally reproduce established MATLAB methods.

Deliverables:

- `src/usctbench/adapters/matlab.py`
- `src/usctbench/algorithms/adapters/refraction_gn.py`
- `src/usctbench/algorithms/adapters/rwave.py`
- `configs/algorithms/bent_ray_gn.yaml`
- `configs/algorithms/rwave_adapter.yaml`
- `docs/algorithm_cards/bent_ray_gn.md`
- `docs/algorithm_cards/rwave_ray_born.md`

Rules:

- Do not rewrite large MATLAB packages blindly.
- Wrap external code as an optional adapter.
- Save input `.mat` files generated for MATLAB under the run directory.
- Save stdout/stderr logs from MATLAB.
- If MATLAB is unavailable, skip with a clear message, not a crash.

### Agent 6 — Tiny FWI lead

Mission: implement a small proof-of-life FWI, not production FWI.

Deliverables:

- `src/usctbench/algorithms/fwi/tiny_fwi.py`
- `src/usctbench/algorithms/fwi/losses.py`
- `src/usctbench/algorithms/fwi/gradient_check.py`
- `configs/algorithms/fwi_tiny.yaml`
- `docs/algorithm_cards/fwi_tiny.md`
- `tests/test_fwi_gradient_check.py`
- `tests/test_fwi_loss_decrease.py`

v0.1 restrictions:

- Start with sound speed only.
- Start with a tiny synthetic case, then OpenBreastUS smoke.
- Use low frequency first.
- Require gradient check on synthetic case.
- Require loss decrease before claiming success.
- Do not add attenuation inversion until sound-speed-only FWI is stable.

### Agent 7 — Documentation/literature lead

Mission: keep the library useful for later researchers.

Deliverables:

- `docs/algorithm_taxonomy.md`
- `docs/ALGORITHM_SETTINGS_TROUBLESHOOTING.md`
- `docs/EXTERNAL_SOURCES_AND_LICENSES.md`
- `docs/references.bib`
- `docs/benchmark_report_template.md`

Each algorithm card must include:

- physical assumption;
- input requirements;
- default settings;
- expected failure modes;
- what to adjust first;
- acceptance tests;
- references and related code.

### Agent 8 — DevOps/A100 lead

Mission: keep the A100 server reproducible.

Deliverables:

- `environment.yml`
- `requirements.txt`
- `.gitignore`
- `.env.example`
- `scripts/setup_workspace.sh`
- `scripts/check_server.sh`
- `scripts/bootstrap_a100.sh`
- `scripts/run_smoke.sh`
- `docs/A100_SERVER_SETUP.md`

Scripts must never require full OpenBreastUS download. They must operate on local paths.

---

## 8. Standard failure-report format

Every failed benchmark run should write:

```markdown
# Failure report

- Algorithm:
- Case id:
- Config:
- Error type: schema / data / numerical / convergence / external-dependency / unknown
- Symptom:
- Most likely causes:
- First three actions to try:
- Logs:
- Plots:
```

---

## 9. Expert troubleshooting rules

When a run fails, do not tune everything randomly.

Ray methods:

1. Check units, sign convention, geometry, receiver ordering, missing-data mask, phase unwrap, and water/reference subtraction.
2. Inspect sinograms before changing the reconstruction algorithm.
3. For SART/SIRT, lower relaxation, increase smoothing, clip SoS bounds, and increase iterations gradually.
4. For CGLS, check operator scaling and regularization before increasing iterations.

Bent-ray GN:

1. Start from a smoothed straight-ray reconstruction.
2. Increase regularization.
3. Add or strengthen line search.
4. Reduce outer iterations if divergence appears early.
5. Check whether ray tracing produces invalid paths.

Ray-Born/Rytov:

1. Use a better background model.
2. Start from lower frequency.
3. Reject low-SNR receivers.
4. Check complex phase convention and reference field alignment.

FWI:

1. Lower starting frequency.
2. Smooth the initial model.
3. Use frequency continuation.
4. Reduce learning rate or step length.
5. Verify source wavelet, receiver geometry, PML, and gradient sign.
6. Require finite-difference gradient check on a tiny synthetic problem.

---

## 10. Definition of Done for v0.1

v0.1 is done when:

1. `pytest -q` passes locally for synthetic/unit tests.
2. `pytest -q` passes on A100.
3. `usct data inspect-openbreastus` runs on A100.
4. `usct data make-smoke` creates a smoke subset on A100.
5. `straight_sart` and `attenuation_sirt` run on at least one smoke case.
6. `fwi_tiny` passes synthetic gradient check and loss-decrease test.
7. At least one bent-ray or ray-Born adapter is callable or explicitly skipped due to missing MATLAB/external dependency with a clear report.
8. Benchmark reports are generated automatically.
9. Algorithm cards exist for every registered algorithm.
10. No data, checkpoints, external full repos, or run outputs are committed.
