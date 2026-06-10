# Development

This document records the release hygiene commands for `usct-benchlab`.

## Format and Lint

```bash
black src tests scripts
ruff check src tests scripts --fix
```

## Compile and Test

```bash
python -m compileall src tests
pytest -q
```

## CLI Smoke

```bash
usct --help
usct list-algorithms
bash scripts/run_smoke.sh
```

## Release Audit

```bash
python scripts/audit_release.py
```

Additional checks:

```bash
git ls-files | grep -E '\.(h5|hdf5|mat|npy|npz|zarr|pt|pth|ckpt|pkl)$' || true
grep -RIn "/home/example\|/Users/example" README.md docs configs scripts src tests .env.example || true
```

The first command should print nothing. The second command is an example path
pattern check; adapt the user names for local audits.

## Git Hygiene

Commit source, configs, tests, docs, scripts, and small README figures. Do not
commit raw datasets, generated benchmark runs, external repositories, or model
checkpoints.
