# v0.1 Readiness Checklist

This checklist describes the release checks for the cleanup branch. It is a
repository readiness checklist, not a claim that every heavy A100 benchmark has
been rerun in the current local checkout.

## Required Local Checks

- `python -m compileall src tests`
- `pytest -q`
- `usct --help`
- `usct list-algorithms`
- `python scripts/audit_v01_readiness.py --root .`
- `git ls-files | grep -E '\.(h5|hdf5|mat|npy|npz|zarr|pt|pth|ckpt|pkl)$' || true`

## Release Mainline

| Track | Suite | Expected role |
| --- | --- | --- |
| Travel-time surrogate | `configs/benchmarks/travel_time_surrogate_main.yaml` | CGLS/SIRT/SART, bent-ray surrogate, rWave surrogate |
| k-Wave FWI | `configs/benchmarks/kwave_fwi_main.yaml` | high-fidelity pure FWI adapter |
| Smoke/unit | `configs/benchmarks/synthetic_smoke.yaml` and local tests | interface and regression checks |

## Diagnostic or Archived Paths

- k-Wave unified ray/rWave feature suites are archived under
  `configs/benchmarks/archive/`.
- External MATLAB adapter suites are archived and optional. They require
  third-party code outside this repository.
- Experimental algorithm configs live under `configs/algorithms/experimental/`.
- Historical FWI alignment notes live under `docs/experiments/`.

## Audit Commands

Repository-only audit:

```bash
python scripts/audit_v01_readiness.py --root .
```

Audit a benchmark run:

```bash
python scripts/audit_v01_readiness.py \
  --root . \
  --run-dir "$USCT_RUN_ROOT/<run_id>"
```

Audit a quality evidence bundle:

```bash
python scripts/audit_v01_readiness.py \
  --root . \
  --quality-evidence-root "$USCT_RUN_ROOT"
```

Full OpenBreastUS smoke evidence, when local data is available:

```bash
python scripts/audit_v01_readiness.py \
  --root . \
  --run-dir "$USCT_RUN_ROOT/<run_id>" \
  --openbreastus-index "$USCT_RUN_ROOT/openbreastus_index.json" \
  --smoke-manifest "$USCT_SAMPLE_ROOT/openbreastus_smoke_manifest.json" \
  --require-v01-dod
```

Convenience release check:

```bash
bash scripts/run_v01_release_check.sh
```

If `USCT_DATA_ROOT` is not present, the convenience script runs repository
checks and prints an explicit data-benchmark skip. Set `USCT_DATA_ROOT` and the
workspace variables documented in `README.md` to run the full data evidence
chain.

## Current Release Evidence Policy

- README comparison panels are small, committed, compressed artifacts under
  `docs/assets/`.
- Raw runs, generated HDF5/MAT files, datasets, checkpoints, and external code
  must remain outside Git.
- Travel-time surrogate metrics should be reported as baseline/sanity evidence.
- FWI metrics should be reported as k-Wave full-wave evidence.
- k-Wave-derived ray/rWave/true-bent results should be reported only as
  diagnostic evidence unless a future release adds separate validation.
