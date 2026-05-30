# v0.1 Readiness Checklist

This checklist records current evidence. It is not a claim that v0.1 is complete.

| Requirement | Current evidence | Status |
|---|---|---|
| Local `pytest -q` passes | Current branch test suite passes locally on `codex-agent0-skeleton` | Passing |
| A100 `pytest -q` passes | Current branch test suite passes with `/home/wudalong/miniconda3/bin/python` | Passing |
| `usct --help` works | editable install and CLI smoke verified locally | Passing |
| OpenBreastUS inspect runs on A100 | writes index for `breast_train_speed.mat` | Passing for current speed-only mirror |
| Smoke subset created on A100 | writes manifest and `cases/breast_train_speed_000000.h5` | Passing |
| `straight_sart` smoke run | A100 smoke benchmark writes standard artifacts and passes thresholds | Passing |
| `attenuation_sirt` smoke run | A100 smoke benchmark writes standard artifacts and passes thresholds using zero surrogate log amplitude | Passing with speed-only surrogate limitation |
| Tiny FWI gradient check | `test_fwi_gradient_check.py` | Passing synthetic-only |
| Tiny FWI loss decrease | `test_fwi_loss_decrease.py` | Passing synthetic-only |
| MATLAB/r-Wave adapter callable or skipped clearly | adapter tests and readiness audit verify `skipped`, external-dependency failure report, and adapter skip metadata | Passing skip path |
| Benchmark reports generated automatically | `usct bench` writes CSV and Markdown with reasons/runtime/memory | Passing |
| Algorithm cards exist for registered algorithms | cards under `docs/algorithm_cards/` | Passing |
| No data/run/checkpoint outputs committed | `.gitignore` plus current branch status | Passing in current commits |

## Known Gaps

- The A100 OpenBreastUS mirror currently exposes speed maps only. Real measured RF/wavefield/reference data is not yet available in the configured root.
- `attenuation_sirt` smoke on the converted speed-only case uses a zero log-amplitude surrogate; it verifies harness plumbing, not physical attenuation recovery.
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

Use `--require-clean` only when intentionally auditing a clean release checkout; Codex working threads may have user-owned unstaged instruction edits.

`scripts/run_smoke.sh` runs `pytest` everywhere. When `$USCT_SAMPLE_ROOT/cases/*.h5` exists, it also runs `configs/benchmarks/openbreastus_smoke.yaml` and audits the generated run directory.
