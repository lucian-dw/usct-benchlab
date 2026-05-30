# usct-benchlab

`usct-benchlab` is a traditional-first benchmark harness for ultrasound computed tomography (USCT) reconstruction algorithms.

The v0.1 scope is deliberately classical: shared `USCTCase -> ReconstructionResult` I/O, OpenBreastUS inspection, ray-based sound-speed and attenuation baselines, metrics, benchmark reports, and optional wrappers for established external methods.

## Quick start

```bash
pip install -e ".[dev]"
usct --help
pytest -q
```

## Current v0.1 commands

```bash
usct data inspect-openbreastus --root "$USCT_DATA_ROOT" --out openbreastus_index.json
usct data make-smoke --root "$USCT_DATA_ROOT" --out "$USCT_SAMPLE_ROOT" --cases-per-density 1
usct list-algorithms
usct run straight_cgls --case "$USCT_SAMPLE_ROOT/cases/<case>.h5" --config configs/algorithms/straight_cgls.yaml --out runs/manual
usct eval --run runs/manual --protocol configs/benchmarks/openbreastus_smoke.yaml
usct bench --suite configs/benchmarks/openbreastus_smoke.yaml
```

Large datasets, generated runs, external repositories, and checkpoints belong outside Git under the workspace-level `data/`, `runs/`, `external/`, and `checkpoints/` directories.

The OpenBreastUS inspector is intentionally schema-first: it writes an index and schema report from the local tree before any loader assumes case layout or filenames. For speed-map-only MATLAB v7.3 mirrors such as `breast_train_speed.mat`, `make-smoke` writes downsampled standard `USCTCase` HDF5 files under `$USCT_SAMPLE_ROOT/cases/` and records the surrogate straight-ray feature assumptions in metadata.

## Documentation

- [Architecture](docs/architecture.md)
- [OpenBreastUS data protocol](docs/OPENBREASTUS_DATA_PROTOCOL.md)
- [Evaluation acceptance protocol](docs/EVALUATION_ACCEPTANCE_PROTOCOL.md)
- [Algorithm taxonomy](docs/algorithm_taxonomy.md)
- [External sources and licenses](docs/EXTERNAL_SOURCES_AND_LICENSES.md)
- [v0.1 readiness checklist](docs/V0_1_READINESS_CHECKLIST.md)
