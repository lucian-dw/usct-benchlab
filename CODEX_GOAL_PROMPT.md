# Codex Goal Mode Prompt

Copy this whole prompt into Codex Goal Mode.

---

You are working on my private GitHub repository:

```text
git@github.com:Math-Wu/usct-benchlab.git
```

Project name: `usct-benchlab`.

This is a USCT traditional algorithm benchmark library for a medical imaging agent. The first version must focus on **traditional and classical algorithms only**. Do **not** implement diffusion, GAN, score model, or heavy generative models in v0.1. Keep extension points for deep learning, but the acceptance target is classical USCT algorithms.

## Context

USCT algorithms should be organized by physical modeling complexity:

1. data/geometric calibration and feature extraction;
2. straight-ray travel-time tomography;
3. straight-ray attenuation tomography;
4. refraction-corrected/bent-ray travel-time tomography;
5. weak-scattering Born/Rytov/ray-Born reconstruction;
6. small full-waveform inversion proof-of-life.

The dataset is OpenBreastUS. It is already downloaded by me on the A100 server. Do not download the full dataset. Inspect the local data path and build smoke/mini subsets.

Expected environment variables:

```bash
export USCT_DATA_ROOT=/data/openbreastus
export USCT_SAMPLE_ROOT=/data/openbreastus_sample
export USCT_RUN_ROOT=/data/usctbench_runs
```

If these paths do not exist, create clear instructions and continue with synthetic fixtures so the repository remains testable.

## Main goal

Build a working v0.1 library where algorithms can be called by an agent through a stable CLI:

```bash
usct data inspect-openbreastus --root $USCT_DATA_ROOT
usct data make-smoke --root $USCT_DATA_ROOT --out $USCT_SAMPLE_ROOT --cases-per-density 1
usct list-algorithms
usct run straight_sart --case <case.h5> --config configs/algorithms/straight_sart.yaml --out <run_dir>
usct eval --run <run_dir> --protocol configs/benchmarks/openbreastus_smoke.yaml
usct bench --suite configs/benchmarks/openbreastus_smoke.yaml
```

## Required deliverables

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

Implement:

1. `USCTCase` and `ReconstructionResult` schema.
2. HDF5 read/write helpers.
3. Algorithm registry.
4. Typer/Rich CLI.
5. OpenBreastUS inspector and smoke subset creator.
6. Feature extraction for ray methods: phase-delay / travel-time-like feature, log-amplitude ratio, and valid mask.
7. Metrics: ROI RMSE, MAE, NRMSE, SSIM, PSNR, runtime, residual, pass/fail report.
8. Straight-ray projector with adjoint test.
9. Straight-ray SART/SIRT/CGLS sound-speed reconstruction.
10. Straight-ray attenuation tomography.
11. Optional MATLAB adapters for refraction-corrected GN and r-Wave ray-Born.
12. Tiny FWI with synthetic gradient check and loss-decrease test.
13. Algorithm cards with settings and troubleshooting.

## External code to study or optionally wrap

Use these as references/adapters, not as uncontrolled code dumps:

- `ucl-bug/ust-sart`: straight-ray SART for ring-array UST from ToF data.
- `rehmanali1994/refractionCorrectedUSCT.github.io`: MATLAB Gauss-Newton travel-time USCT with Laplacian/Bayesian/resolution-filling variants.
- `Ash1362/ray-based-quantitative-ultrasound-tomography`: r-Wave MATLAB package for quantitative USCT via ray-Born inversion.
- `rehmanali1994/WaveformInversionUST`: frequency-domain waveform inversion for ring-array UST.
- `rehmanali1994/FrequencyDifferencing`: low-frequency synthesis to mitigate FWI cycle skipping.
- `ucl-bug/k-wave`, `k-wave-python`, or `deepwave`: optional wave simulation/FWI backends.
- `astra-toolbox`: optional accelerated tomography primitives.

Check licenses before vendoring. Prefer `external/` adapters and documented install steps.

## Implementation order

Do not start with FWI. Follow this order:

1. Repo skeleton, package install, tests, CLI.
2. Schema and HDF5 roundtrip.
3. OpenBreastUS inspection and smoke subset creation.
4. Metrics and benchmark runner.
5. Straight-ray projector and SART/SIRT/CGLS.
6. Attenuation tomography.
7. Synthetic fixtures and strict tests.
8. Optional MATLAB adapters.
9. Tiny FWI proof-of-life.
10. Documentation and benchmark report.

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
- synthetic correctness test passes;
- OpenBreastUS smoke run produces metrics;
- failure report is generated if convergence fails.

Target initial thresholds are engineering smoke thresholds, not final paper claims:

- straight-ray SART should improve over water/background baseline on smoke cases;
- attenuation tomography should reduce log-amplitude residual on synthetic and smoke feature tests;
- tiny FWI should pass finite-difference gradient check on synthetic data and reduce waveform residual;
- bent-ray/ray-Born adapters may be optional but must be callable and documented.

## Expert troubleshooting rules

When a run fails, do not randomly tune everything. Follow the expert guide in `docs/ALGORITHM_SETTINGS_TROUBLESHOOTING.md`:

- For ray methods, first check units, sign convention, geometry, missing-data mask, phase unwrap, and water/reference subtraction.
- For SART/SIRT, lower relaxation, increase smoothing, clip SoS bounds, and inspect the sinogram before changing the algorithm.
- For bent-ray GN, start from a smoothed straight-ray result, increase regularization, add line search, and reduce outer iterations.
- For ray-Born/Rytov, use a better background, lower frequency first, and reject low-SNR receivers.
- For FWI, lower the starting frequency, smooth the initial model, use frequency continuation, reduce learning rate, verify source wavelet/PML, and check gradient sign.

## Server behavior

Before heavy work, run:

```bash
pwd
nvidia-smi
python --version
git status
```

Do not assume CUDA version. Use CPU-compatible tests where possible and GPU only for benchmark/FWI. A100 memory is large, but v0.1 should still support small smoke runs.

## Final output expected from you

When you finish, provide:

1. a concise summary of implemented components;
2. exact commands to run smoke benchmark;
3. known limitations;
4. which algorithms passed/failed and why;
5. a commit hash or branch name.

Begin by setting up the repository skeleton and tests. Work incrementally and commit often.

---
