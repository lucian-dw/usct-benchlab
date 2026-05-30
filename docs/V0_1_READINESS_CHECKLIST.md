# v0.1 Readiness Checklist

This checklist records current evidence. It is not a claim that v0.1 is complete.

| Requirement | Current evidence | Status |
|---|---|---|
| Local `pytest -q` passes | Current branch test suite passes locally on `codex-agent0-skeleton` | Passing |
| A100 `pytest -q` passes | Current branch test suite passes with `/home/wudalong/miniconda3/bin/python` | Passing |
| `usct --help` works | editable install and CLI smoke verified locally | Passing |
| OpenBreastUS inspect runs on A100 | writes index for `breast_train_speed.mat` plus linked local k-Wave simulation MAT evidence | Passing for current A100 mirror |
| Smoke subset created on A100 | writes manifest and `cases/a_kwave_train_6602.h5` after clearing stale converted cases | Passing |
| `straight_sart` smoke run | A100 smoke benchmark writes standard artifacts and passes thresholds | Passing |
| `attenuation_sirt` smoke run | A100 smoke benchmark writes standard artifacts and passes thresholds using simulated nonzero attenuation evidence from a k-Wave channel MAT file | Passing with simulated-data limitation |
| Tiny FWI gradient check | `test_fwi_gradient_check.py` | Passing synthetic-only |
| Tiny FWI loss decrease | `test_fwi_loss_decrease.py` | Passing synthetic-only |
| MATLAB/r-Wave adapter callable or skipped clearly | adapter tests and readiness audit verify `skipped`, external-dependency failure report, and adapter skip metadata | Passing skip path |
| Benchmark reports generated automatically | `usct bench` writes CSV and Markdown with reasons/runtime/memory | Passing |
| Algorithm cards exist for registered algorithms | cards under `docs/algorithm_cards/` | Passing |
| No data/run/checkpoint outputs committed | `.gitignore` plus current branch status | Passing in current commits |

## Known Gaps

- The configured A100 data root includes a local symlink to a k-Wave simulation MAT file so attenuation smoke has nonzero simulated attenuation evidence. It is not raw measured OpenBreastUS RF data.
- The speed-map-only `breast_train_speed.mat` path remains supported, but zero `log_amp` surrogate cases no longer count as valid attenuation DoD evidence.
- MATLAB adapters do not yet marshal `USCTCase` into external package input formats.
- `fwi_tiny` is synthetic proof-of-life only and is not production FWI.

## Audit Command

Repository readiness can be checked with:

```bash
python scripts/audit_v01_readiness.py --root .
```

To include benchmark evidence from A100:

```bash
python scripts/audit_v01_readiness.py --root . --run-dir "$USCT_RUN_ROOT/<run_id>"
```

For v0.1 release evidence, include the index and smoke manifest:

```bash
python scripts/audit_v01_readiness.py \
  --root . \
  --run-dir "$USCT_RUN_ROOT/<run_id>" \
  --openbreastus-index "$USCT_RUN_ROOT/openbreastus_index.json" \
  --smoke-manifest "$USCT_SAMPLE_ROOT/openbreastus_smoke_manifest.json" \
  --require-v01-dod
```

The full v0.1 evidence chain can be run on A100 with:

```bash
PYTHON_BIN=/home/wudalong/miniconda3/bin/python bash scripts/run_v01_release_check.sh
```

This script runs `pytest`, `inspect-openbreastus`, `make-smoke`, the smoke
benchmark suite, and `audit_v01_readiness.py --require-v01-dod` with explicit
OpenBreastUS index, smoke manifest, and run-directory evidence.

The latest verified A100 release check in this thread used:

```text
run_root=/home/wudalong/usct-benchlab/runs/usctbench_runs/openbreastus_smoke_20260530T152151Z
git_commit=f20d5542e8584c0b50bce3acb94d8013c824ec17
records=2
passed=2
```

Use `--require-clean` only when intentionally auditing a clean release checkout; Codex working threads may have user-owned unstaged instruction edits.

`scripts/run_smoke.sh` runs `pytest` everywhere. When `$USCT_SAMPLE_ROOT/cases/*.h5` exists, it also runs `configs/benchmarks/openbreastus_smoke.yaml` and audits the generated run directory.
