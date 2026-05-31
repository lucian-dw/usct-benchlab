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
| Bent-ray native backend | OpenBreastUS and NBPslice2D smoke/quality runs write standard metrics and previews; optional MATLAB path still skips clearly when requested | Passing native v0.1 path |
| r-Wave native backend | OpenBreastUS and NBPslice2D smoke/quality runs write standard metrics and previews; optional MATLAB path still skips clearly when requested | Passing native v0.1 path |
| k-Wave FWI adapter | A100 `fwi_kwave_full_pipeline_success201_dense` and re-ingest evidence show external result loading, GT/native metrics, and visual artifacts | Passing one-case A100 smoke |
| MATLAB adapter dependency path | adapter tests and readiness audit verify `skipped`, external-dependency failure report, and adapter skip metadata when MATLAB backend is requested and unavailable | Passing skip path |
| Benchmark reports generated automatically | `usct bench` writes CSV and Markdown with reasons/runtime/memory | Passing |
| Algorithm cards exist for registered algorithms | cards under `docs/algorithm_cards/` | Passing |
| No data/run/checkpoint outputs committed | `.gitignore` plus current branch status | Passing in current commits |

## Known Gaps

- The configured A100 data root includes a local symlink to a k-Wave simulation MAT file so attenuation smoke has nonzero simulated attenuation evidence. It is not raw measured OpenBreastUS RF data.
- The speed-map-only `breast_train_speed.mat` path remains supported, but zero `log_amp` surrogate cases no longer count as valid attenuation DoD evidence.
- MATLAB adapters now export a standard MATLAB-readable `USCTCase` input MAT
  file, execute configured entrypoints with `usctbench_input_mat` and
  `usctbench_output_mat` variables, and ingest standard adapter output MAT
  files. The default `bent_ray_gn` and `rwave_adapter` paths remain native
  benchmark backends; public-package-specific entrypoint scripts still need to
  be authored outside Git for full end-to-end external runs.
- `fwi_tiny` is synthetic proof-of-life only. Production-like FWI evidence is
  currently the A100 k-Wave adapter path, not `fwi_tiny`.
- k-Wave FWI has one validated OpenBreastUS test201 smoke case. It is not yet a
  broad multi-case benchmark.

## Audit Command

Repository readiness can be checked with:

```bash
python scripts/audit_v01_readiness.py --root .
```

To include benchmark evidence from A100:

```bash
python scripts/audit_v01_readiness.py --root . --run-dir "$USCT_RUN_ROOT/<run_id>"
```

To include the current 256 quality and FWI visual evidence bundle from A100:

```bash
python scripts/audit_v01_readiness.py \
  --root . \
  --quality-evidence-root "$USCT_RUN_ROOT"
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

`scripts/run_openbreastus_smoke.sh` is the explicit A100 OpenBreastUS smoke
entrypoint when the sample set has not yet been generated. It inspects the data
root, creates 64x64 smoke cases, runs `straight_sart`, `bent_ray_gn`,
`rwave_adapter`, and `attenuation_sirt`, then renders a grayscale sound-speed
comparison panel for the three sound-speed algorithms.

## Current Traditional/FWI Evidence Paths

Representative A100 evidence from the current branch:

- OpenBreastUS 4-class 256 quality:
  `/home/wudalong/usct-benchlab/runs/usctbench_runs/openbreastus_quality_20260531T164948Z`
  - run check: `benchmark_run_checks.json`
  - panel: `comparison_artifacts/openbreastus_quality_256_4class_5alg_gray.png`
- NBPslice2D 4-class 256 quality:
  `/home/wudalong/usct-benchlab/runs/usctbench_runs/nbpslice2d_quality_20260531T162341Z`
  - run check: `benchmark_run_checks.json`
  - panel: `comparison_artifacts/nbpslice2d_quality_256_4class_5alg_gray.png`
- k-Wave FWI successful full-pipeline result:
  `/home/wudalong/usct-benchlab/runs/usctbench_runs/fwi_kwave_full_pipeline_success201_dense`
- k-Wave FWI re-ingest with corrected acceptance metrics:
  `/home/wudalong/usct-benchlab/runs/usctbench_runs/fwi_kwave_success201_reingest_63ec1e5`
  - run check: `benchmark_run_checks.json`
  - contact sheet: `fwi_kwave_adapter/openbreast_test201_kwave_full_000200/kwave_smoke_outputs/contact_sheet.png`
- Single-case cross-algorithm visual comparison:
  `/home/wudalong/usct-benchlab/runs/usctbench_runs/fwi_kwave_cross_algorithm_63ec1e5/comparison_artifacts/fwi_case_test201_cross_algorithm_horizontal_gray.png`

The current evidence bundle can be audited on A100 with:

```bash
python scripts/audit_v01_readiness.py \
  --root /home/wudalong/usct-benchlab/code \
  --quality-evidence-root /home/wudalong/usct-benchlab/runs/usctbench_runs
```
