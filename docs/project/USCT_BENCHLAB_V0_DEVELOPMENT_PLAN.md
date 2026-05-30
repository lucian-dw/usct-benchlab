# `usct-benchlab` v0.1 Development Plan — Traditional Classical Algorithms First

Date: 2026-05-30

Private repo:

```text
git@github.com:Math-Wu/usct-benchlab.git
```

## 1. v0.1 thesis

The first release should not try to be a “large AI reconstruction zoo.” It should be a clean, trustworthy, traditional USCT algorithm benchmark library. The contribution is:

> a physically organized USCT algorithm stack with unified I/O, OpenBreastUS smoke/mini benchmarks, explicit acceptance tests, and expert troubleshooting notes.

This is more useful than a folder of scripts because a medical-imaging agent can call algorithms consistently and know whether they actually ran successfully.

## 2. Algorithm scope

### Included in v0.1

| Layer | Algorithm family | Implementation target | Why include |
|---|---|---|---|
| A0 | data inspection and feature extraction | Python | Required for all algorithms |
| A1 | straight-ray ToF tomography | native Python SART/SIRT/CGLS + reference to `ust-sart` | simplest quantitative SoS baseline |
| A1b | straight-ray attenuation tomography | native Python SIRT/CGLS | simple quantitative attenuation baseline |
| A2 | refraction-corrected travel-time tomography | MATLAB adapter or minimal GN | classic improvement over straight-ray |
| A3 | weak-scattering ray-Born/Rytov | r-Wave adapter first | bridge between ray methods and FWI |
| A4 | tiny FWI proof-of-life | native small differentiable implementation or Deepwave backend | verify full-wave direction without huge engineering burden |

### Excluded from v0.1 acceptance

- Diffusion models.
- Score-based posterior sampling.
- GANs.
- Large neural operators.
- Heavy supervised CNN training.

Keep `src/usctbench/algorithms/learning/` as a placeholder only.

## 3. Repository structure

```text
usct-benchlab/
  pyproject.toml
  README.md
  AGENTS.md
  .gitignore
  .env.example
  environment.yml
  requirements.txt

  src/usctbench/
    __init__.py
    schema.py
    registry.py
    cli.py
    io/
      hdf5.py
      matlab.py
    data/
      openbreastus.py
      features.py
      smoke_subset.py
      synthetic.py
    algorithms/
      base.py
      ray/
        straight_projector.py
        sart.py
        sirt.py
        cgls.py
        attenuation.py
      adapters/
        matlab.py
        refraction_gn.py
        rwave.py
      fwi/
        tiny_fwi.py
        losses.py
        gradient_check.py
      learning/
        README.md
    metrics/
      image.py
      data_consistency.py
    benchmark/
      runner.py
      report.py
    viz/
      preview.py

  configs/
    algorithms/
    benchmarks/

  docs/
    algorithm_cards/
    architecture.md
    algorithm_taxonomy.md
    ALGORITHM_SETTINGS_TROUBLESHOOTING.md
    OPENBREASTUS_DATA_PROTOCOL.md
    EVALUATION_ACCEPTANCE_PROTOCOL.md
    EXTERNAL_SOURCES_AND_LICENSES.md

  scripts/
    check_server.sh
    bootstrap_a100.sh
    run_smoke.sh

  tests/
    test_schema_roundtrip.py
    test_projector_adjoint.py
    test_straight_ray_synthetic.py
    test_attenuation_synthetic.py
    test_metrics.py
    test_fwi_gradient_check.py
```

## 4. Unified I/O design

### `USCTCase`

The case object should store everything an algorithm needs, without forcing every algorithm to use every field.

Required conceptual fields:

```yaml
case_id: string
grid:
  shape: [ny, nx]
  spacing_m: [dy, dx]
  origin_m: [y0, x0]
  roi_mask: optional bool[ny, nx]
geometry:
  type: ring | linear | custom
  tx_pos_m: float[ntx, 2]
  rx_pos_m: float[nrx, 2]
  radius_m: optional float
measurement:
  domain: frequency | time | features
  frequencies_hz: optional float[nf]
  freq_data: optional complex[nf, ntx, nrx]
  time_data: optional float[ntx, nrx, nt]
  tof_s: optional float[ntx, nrx]
  delta_tof_s: optional float[ntx, nrx]
  log_amp: optional float[nf or 1, ntx, nrx]
  valid_mask: optional bool[...]
ground_truth:
  sound_speed_mps: optional float[ny, nx]
  attenuation_np_per_m: optional float[ny, nx]
metadata: dict
```

### `ReconstructionResult`

```yaml
algorithm: string
case_id: string
sound_speed_mps: optional float[ny, nx]
attenuation_np_per_m: optional float[ny, nx]
reflectivity: optional float[ny, nx]
uncertainty: optional float[ny, nx]
metrics: dict
runtime_s: float
status: success | failed | skipped
failure_reason: optional string
artifacts: dict
```

## 5. Milestones

### Milestone 0 — repository skeleton

Acceptance:

- package installs with `pip install -e .`;
- `usct --help` works;
- `pytest -q` runs;
- `.gitignore` protects data/checkpoints/runs.

### Milestone 1 — schema and synthetic fixtures

Acceptance:

- `USCTCase` HDF5 roundtrip test passes;
- synthetic homogeneous and circular-inclusion phantoms can be generated;
- straight-ray feature generation works on synthetic phantoms.

### Milestone 2 — OpenBreastUS inspection

Acceptance:

- `usct data inspect-openbreastus --root $USCT_DATA_ROOT` writes an index;
- smoke subset builder creates 1 case per density type if available;
- feature extraction writes diagnostic images and masks.

### Milestone 3 — metrics/benchmark

Acceptance:

- `usct eval` writes `metrics.json`;
- `usct bench` writes `benchmark_summary.csv` and `benchmark_report.md`;
- failing runs get `failure_report.md`.

### Milestone 4 — straight-ray sound speed

Acceptance:

- projector adjoint test relative error `<1e-4`;
- homogeneous synthetic RMSE `<2 m/s`;
- circular synthetic reconstruction improves over water baseline;
- OpenBreastUS smoke run produces finite SoS map and metrics.

### Milestone 5 — attenuation tomography

Acceptance:

- synthetic attenuation reconstruction reduces log-amplitude residual;
- OpenBreastUS smoke feature run produces finite attenuation map;
- algorithm card documents that this is a first-order baseline.

### Milestone 6 — adapters

Acceptance:

- MATLAB wrapper can call an external script if MATLAB exists;
- if MATLAB is missing, adapter returns `skipped` with clear reason;
- generated `.mat` input and converted `.h5` output are logged.

### Milestone 7 — tiny FWI

Acceptance:

- gradient finite-difference check relative error `<5e-2` on tiny synthetic case;
- waveform/data residual decreases by at least 30% on tiny case;
- OpenBreastUS smoke run is attempted only after synthetic tests pass.

## 6. v0.1 benchmark suites

### `openbreastus_smoke`

Purpose: verify code path.

- 4 cases if possible: HET, FIB, FAT, EXD.
- 1 frequency first: 300 kHz or the lowest available frequency.
- Subsample transmitters/receivers for fast runtime.
- Downsample grid to 128×128 or smaller if needed.

### `openbreastus_mini`

Purpose: compare algorithms more meaningfully.

- 32–80 cases.
- 3 frequencies: low, middle, high.
- more transmitters/receivers.
- stratified by density class.

## 7. Algorithm cards

Every algorithm card must answer:

1. What physics does this algorithm assume?
2. What data does it require?
3. What is the default config?
4. What can go wrong?
5. What should be adjusted first?
6. What metrics define pass/fail?
7. Which papers/code inspired the implementation?

## 8. Why this is a real contribution

For USCT newcomers, the hard part is not only implementing FWI. It is understanding when each physical approximation is appropriate and how the reconstruction pipeline fails. This repository should become a practical map:

- straight-ray = fastest sanity baseline;
- attenuation = separate quantitative contrast baseline;
- bent-ray = refraction correction when ToF is reliable;
- ray-Born/Rytov = weak-scattering bridge to wave methods;
- FWI = high-resolution but sensitive to initialization, frequency, calibration, and cycle skipping.
