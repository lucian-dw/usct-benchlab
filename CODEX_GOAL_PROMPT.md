# CODEX_GOAL_PROMPT.md — Start `usct-benchlab` v0.1

Copy this prompt into Codex Goal Mode.

---

You are working on my private GitHub repository:

```text
git@github.com:Math-Wu/usct-benchlab.git
```

Project name: `usct-benchlab`.

Read these files first, in order:

1. `AGENTS.md`
2. `README.md` if it exists
3. `CODEX_TASKS.md` if it exists
4. `docs/project/USCT_BENCHLAB_V0_DEVELOPMENT_PLAN.md` if it exists
5. `docs/project/ALGORITHM_SETTINGS_TROUBLESHOOTING.md` if it exists
6. `docs/setup/A100_SERVER_SETUP.md` if it exists

## Current real workflow

I use Codex locally on my Mac, but the Mac is only a lightweight editing/control environment. The A100 server is the authoritative runtime environment for CUDA, PyTorch-GPU, MATLAB/external wrappers, OpenBreastUS inspection, and benchmark runs.

Local Mac workspace:

```text
/Users/wudalong/Desktop/usct-benchlab
```

A100 workspace:

```text
~/usct-benchlab
```

The Git repository may be either the workspace root or a `code/` subdirectory. Detect the repository root using:

```bash
git rev-parse --show-toplevel
```

Use GitHub as the synchronization bridge:

```text
Mac edit -> git commit/push -> GitHub -> A100 git pull -> A100 run tests/benchmarks
```

If you need to run heavy commands, use the configured SSH MCP connection to A100. Do not assume Codex is running directly on A100.

Before any A100 run, execute remotely:

```bash
cd ~/usct-benchlab/code 2>/dev/null || cd ~/usct-benchlab
git status
git pull --ff-only || git pull
bash scripts/setup_workspace.sh || true
nvidia-smi
python --version
```

Do not use `/data/...`. I do not have sudo permission on the A100 server. Use workspace-relative paths:

```bash
export USCT_WORKSPACE=$HOME/usct-benchlab
export USCT_DATA_ROOT=$USCT_WORKSPACE/data/openbreastus
export USCT_SAMPLE_ROOT=$USCT_WORKSPACE/data/openbreastus_sample
export USCT_RUN_ROOT=$USCT_WORKSPACE/runs/usctbench_runs
```

On Mac, do not require CUDA, MATLAB, k-Wave, Deepwave, or the full OpenBreastUS dataset. Mac tests should be limited to static checks, schema tests, CLI import tests, and tiny synthetic fixtures.

## Project goal

This is a USCT traditional algorithm benchmark library for a medical imaging agent. v0.1 must focus on **traditional and classical algorithms only**.

Do **not** implement diffusion, GAN, score-based models, large neural-operator training, or heavy generative models in v0.1. Keep extension points for learning methods, but the v0.1 acceptance target is classical USCT algorithms.

Organize USCT algorithms by physical modeling complexity:

1. data/geometric calibration and feature extraction;
2. straight-ray travel-time tomography;
3. straight-ray attenuation tomography;
4. refraction-corrected/bent-ray travel-time tomography;
5. weak-scattering Born/Rytov/ray-Born reconstruction;
6. tiny full-waveform inversion proof-of-life.

The dataset is OpenBreastUS. It is already downloaded by me on A100. Do not download the full dataset. Inspect the local data path and build smoke/mini subsets.

## Main deliverable

Build a working v0.1 library where algorithms can be called by an agent through a stable CLI:

```bash
usct data inspect-openbreastus --root $USCT_DATA_ROOT
usct data make-smoke --root $USCT_DATA_ROOT --out $USCT_SAMPLE_ROOT --cases-per-density 1
usct list-algorithms
usct run straight_sart --case <case.h5> --config configs/algorithms/straight_sart.yaml --out <run_dir>
usct eval --run <run_dir> --protocol configs/benchmarks/openbreastus_smoke.yaml
usct bench --suite configs/benchmarks/openbreastus_smoke.yaml
```

Create or update:

```text
pyproject.toml
README.md
AGENTS.md
.gitignore
.env.example
environment.yml
requirements.txt
src/usctbench/
configs/algorithms/
configs/benchmarks/
docs/
scripts/
tests/
```

## Required implementation order

Do not start with FWI. Follow this order:

1. Repository skeleton, package install, tests, CLI.
2. `USCTCase` and `ReconstructionResult` schema.
3. HDF5 read/write helpers.
4. Algorithm registry.
5. `scripts/setup_workspace.sh` that works both on Mac and A100.
6. OpenBreastUS inspector and smoke subset creator.
7. Feature extraction for ray methods:
   - phase-delay / travel-time-like feature;
   - log-amplitude ratio;
   - valid mask;
   - water/reference handling when available.
8. Metrics: ROI RMSE, MAE, NRMSE, SSIM, PSNR, runtime, residual, pass/fail report.
9. Straight-ray projector with adjoint test.
10. Straight-ray SART/SIRT/CGLS sound-speed reconstruction.
11. Straight-ray attenuation tomography.
12. Synthetic fixtures and strict tests.
13. Optional MATLAB adapters for refraction-corrected GN and r-Wave/ray-Born.
14. Tiny FWI with synthetic gradient check and loss-decrease test.
15. Algorithm cards with settings and troubleshooting.

## External code to study or optionally wrap

Use these as references/adapters, not as uncontrolled code dumps:

- `ucl-bug/ust-sart`: straight-ray SART for ring-array UST from ToF data.
- `rehmanali1994/refractionCorrectedUSCT.github.io`: MATLAB Gauss-Newton travel-time USCT with Laplacian/Bayesian/resolution-filling variants.
- `Ash1362/ray-based-quantitative-ultrasound-tomography`: r-Wave MATLAB package for quantitative USCT via ray-Born inversion.
- `rehmanali1994/WaveformInversionUST`: frequency-domain waveform inversion for ring-array UST.
- `rehmanali1994/FrequencyDifferencing`: low-frequency synthesis to mitigate FWI cycle skipping.
- `ucl-bug/k-wave`, `k-wave-python`, or `deepwave`: optional wave simulation/FWI backends.
- `astra-toolbox`: optional accelerated tomography primitives.

Check licenses before vendoring. Prefer adapters and documented install steps. Do not commit full external repos unless explicitly approved.

## Acceptance rules

An algorithm is not accepted unless it writes:

```text
runs/<run_id>/<case_id>/result.h5
runs/<run_id>/<case_id>/metrics.json
runs/<run_id>/<case_id>/metadata.yaml
runs/<run_id>/<case_id>/preview.png
```

Hard gates:

- no NaN/Inf in ROI;
- output shape matches case grid;
- sound speed in m/s and plausible range;
- all unit tests pass;
- synthetic correctness tests pass;
- OpenBreastUS smoke run produces metrics on A100;
- failure report is generated if convergence fails.

Target initial thresholds are engineering smoke thresholds, not final paper claims:

- straight-ray SART should improve over water/background baseline on smoke cases;
- attenuation tomography should reduce log-amplitude residual on synthetic and smoke feature tests;
- tiny FWI should pass finite-difference gradient check on synthetic data and reduce waveform residual;
- bent-ray/ray-Born adapters may be optional but must be callable or skipped with clear explanation.

## Expert troubleshooting rules

When a run fails, do not randomly tune everything. Follow `docs/project/ALGORITHM_SETTINGS_TROUBLESHOOTING.md` if present, and use this order:

1. Check units, sign convention, geometry, receiver ordering, missing-data mask, phase unwrap, and water/reference subtraction.
2. For SART/SIRT, lower relaxation, increase smoothing, clip SoS bounds, inspect the sinogram, then increase iterations.
3. For bent-ray GN, start from a smoothed straight-ray result, increase regularization, add line search, and reduce outer iterations.
4. For ray-Born/Rytov, use a better background, lower frequency first, reject low-SNR receivers, and verify complex phase convention.
5. For FWI, lower starting frequency, smooth the initial model, use frequency continuation, reduce learning rate, verify source wavelet/PML, and check gradient sign.

## Git and execution rules

- Do not commit data, runs, checkpoints, external cloned repositories, or large scientific files.
- Prefer small commits.
- Before heavy A100 execution, make sure A100 has pulled the latest GitHub state.
- If you patch directly on A100, commit and push from A100, then make sure the Mac pulls the changes.
- Avoid simultaneous uncommitted edits on Mac and A100.
- If data paths are missing, keep the repo testable using synthetic fixtures and write clear instructions instead of failing silently.

## Final output expected from you

When you finish a milestone, provide:

1. concise summary of implemented components;
2. exact commands to run local lightweight tests;
3. exact commands to run A100 smoke benchmark;
4. known limitations;
5. which algorithms passed/failed and why;
6. commit hash or branch name.

Begin now by setting up the repository skeleton, workspace script, tests, and CLI. Then push to GitHub and run the first A100 smoke checks through SSH MCP.
