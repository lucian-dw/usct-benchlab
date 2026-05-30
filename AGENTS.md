# AGENTS.md — Codex multi-agent instructions for `usct-benchlab`

This repository is a USCT traditional algorithm benchmark library. v0.1 is **traditional-first**: do not implement diffusion, GAN, score model, or large neural operator training as a first milestone. Keep the architecture extensible for learning methods, but deliver a robust classical benchmark harness first.

## Non-negotiable constraints

1. **No data in git.** Never commit OpenBreastUS files, checkpoints, generated wavefields, raw `.mat` data, `.h5` cases, or benchmark runs.
2. **Use the private remote**: `git@github.com:Math-Wu/usct-benchlab.git`.
3. **Every algorithm must use the same I/O**: `USCTCase -> ReconstructionResult`.
4. **Every runnable algorithm must have tests, metrics, and an algorithm card.**
5. **Prefer small, verified steps**. Do not build a huge FWI system before the loader, projector, and metrics pass.
6. **Do not silently change units.** Sound speed is `m/s`; slowness is `s/m`; attenuation is `Np/m` unless explicitly converted; frequency is `Hz`; coordinates are meters.
7. **All external code must be isolated** under `external/` or wrapped by an adapter. Check license before vendoring code. Prefer submodules or documented install steps.
8. **OpenBreastUS is already downloaded by the user**. Do not download the full dataset. Inspect and index the existing local copy.
9. **If OpenBreastUS structure differs from assumptions**, write a schema-inspection report and adapt the loader; do not hard-code guessed paths.
10. **If an experiment fails**, write the failure and next actions to `runs/<run_id>/failure_report.md` instead of hiding it.

## Branch and commit discipline

Use short feature branches:

```bash
git checkout -b feat/schema-loader
git checkout -b feat/straight-ray-sart
git checkout -b feat/metrics-benchmark
git checkout -b feat/fwi-tiny
```

Commit only source, configs, tests, docs, and small synthetic fixtures. Do not commit files under:

```text
data/raw/
data/processed/
runs/
checkpoints/
external/*  # if cloned repos are large or license is unclear
*.mat
*.h5
*.npy
*.npz
*.pt
*.pth
*.ckpt
```

## Agent 0 — Architecture lead

Mission: create the package skeleton and the registry that all other agents use.

Deliverables:

- `pyproject.toml`
- `src/usctbench/`
- `src/usctbench/schema.py`
- `src/usctbench/registry.py`
- `src/usctbench/cli.py`
- `src/usctbench/io/hdf5.py`
- `tests/test_schema_roundtrip.py`
- `docs/architecture.md`

Required interfaces:

```python
class USCTCase(BaseModel): ...
class ReconstructionResult(BaseModel): ...
class AlgorithmConfig(BaseModel): ...
class Algorithm(Protocol):
    name: str
    def run(self, case: USCTCase, config: AlgorithmConfig) -> ReconstructionResult: ...
```

CLI must expose:

```bash
usct data inspect-openbreastus
usct data make-smoke
usct run <algorithm>
usct eval --run <run_dir>
usct bench --suite <yaml>
usct list-algorithms
```

## Agent 1 — OpenBreastUS data lead

Mission: turn local OpenBreastUS data into a stable benchmark subset.

Deliverables:

- `src/usctbench/data/openbreastus.py`
- `src/usctbench/data/features.py`
- `src/usctbench/data/smoke_subset.py`
- `data/README.md`
- `docs/OPENBREASTUS_DATA_PROTOCOL.md`
- `tests/test_openbreastus_inspection.py`

Tasks:

1. Inspect the actual local tree at `$USCT_DATA_ROOT`.
2. Generate `openbreastus_index.json` with case id, density class, speed file path, wavefield paths, available frequencies, shape, and split.
3. Create `openbreastus-smoke-v1`: one small case from each density class if available.
4. Convert each selected case to a standard `USCTCase` HDF5 file.
5. Extract robust features for ray algorithms:
   - phase delay / travel-time-like feature;
   - log-amplitude ratio;
   - valid receiver mask;
   - water/reference field handling if present.

Do not assume the phase-delay formula is always correct. Save intermediate diagnostic plots and masks.

## Agent 2 — Metrics and benchmark lead

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

## Agent 3 — Straight-ray tomography lead

Mission: implement the reliable baseline.

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

## Agent 4 — Attenuation tomography lead

Mission: implement a simple but useful attenuation baseline.

Deliverables:

- `src/usctbench/algorithms/ray/attenuation.py`
- `configs/algorithms/attenuation_sirt.yaml`
- `docs/algorithm_cards/attenuation_tomography.md`
- `tests/test_attenuation_synthetic.py`

Use log-amplitude ratios and robust clipping. Treat attenuation as a separate baseline; do not couple it to FWI in v0.1.

## Agent 5 — Refraction and ray-Born lead

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
- Save all input `.mat` files generated for MATLAB under the run directory.
- Save stdout/stderr logs from MATLAB.
- If MATLAB is unavailable, skip with a clear message, not a crash.

## Agent 6 — Tiny FWI lead

Mission: implement a small proof-of-life FWI, not a production FWI.

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

## Agent 7 — Documentation/literature lead

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

## Agent 8 — DevOps/A100 lead

Mission: keep the A100 server reproducible.

Deliverables:

- `environment.yml`
- `requirements.txt`
- `.gitignore`
- `.env.example`
- `scripts/check_server.sh`
- `scripts/bootstrap_a100.sh`
- `scripts/run_smoke.sh`
- `docs/A100_SERVER_SETUP.md`

The scripts must never require full OpenBreastUS download. They must operate on local paths.

## Standard failure-report format

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

## Definition of Done for v0.1

v0.1 is done when:

1. `pytest -q` passes.
2. `usct data inspect-openbreastus` runs on the A100 server.
3. `usct data make-smoke` creates a smoke subset.
4. `straight_sart`, `attenuation_sirt`, and `fwi_tiny` run on at least one smoke case.
5. At least one bent-ray or ray-Born adapter is callable, even if marked optional due to MATLAB/external dependency.
6. Benchmark reports are generated automatically.
7. Algorithm cards exist for every registered algorithm.
8. No data, checkpoints, or run outputs are committed.
