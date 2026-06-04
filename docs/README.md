# usct-benchlab Documentation

This directory separates release documentation from historical experiments.

## Release Docs

- `project/BENCHMARK_TIERS.md`: v0.1 benchmark tracks and what is ranked.
- `project/MEASUREMENT_PROVENANCE.md`: accepted provenance labels.
- `project/SIMULATION_QC.md`: k-Wave simulation QC contract.
- `project/STAGE_SUMMARY.md`: release-stage summary.
- `project/OBSERVABLE_MISMATCH.md`: why k-Wave-derived ray/rWave paths are
  diagnostic-only in v0.1.
- `algorithm_cards/`: release-facing algorithm cards.
- `algorithms/`: implementation notes for selected algorithms.

## Historical Experiments

`experiments/` contains notes from exploratory work and A100 alignment. These
files may mention local example paths or historical run IDs, but they should
not define the release mainline.

## Assets

`assets/` contains small README-ready figures only. Raw runs, `.mat`, `.h5`,
`.npy`, checkpoints, and external code remain outside Git.
