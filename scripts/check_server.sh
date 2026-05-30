#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "$REPO_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$REPO_DIR/.env"
  set +a
fi

cd "$REPO_DIR"

PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-}}"
if [ -z "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python || command -v python3 || true)"
fi

echo "REPO_DIR=$REPO_DIR"
echo "HOSTNAME=$(hostname)"
echo "USER=${USER:-unknown}"
echo "PYTHON_BIN=${PYTHON_BIN:-not-found}"
if [ -n "$PYTHON_BIN" ]; then
  "$PYTHON_BIN" --version || true
fi

echo
echo "Git status:"
git status --short --branch || true

echo
echo "GPU:"
nvidia-smi || true

echo
echo "Python imports:"
if [ -n "$PYTHON_BIN" ]; then
"$PYTHON_BIN" - <<'PY'
import importlib
import sys
print("executable", sys.executable)
for name in ("numpy", "h5py", "pydantic", "yaml", "pytest"):
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        print(f"{name}: ok {version}")
    except Exception as exc:
        print(f"{name}: failed {type(exc).__name__}: {exc}")
try:
    import torch
    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("torch_device:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch: optional import failed", repr(exc))
PY
else
  echo "No Python executable found."
fi

echo
echo "Workspace paths:"
for path in \
  "${USCT_WORKSPACE:-$HOME/usct-benchlab}" \
  "${USCT_DATA_ROOT:-$HOME/usct-benchlab/data/openbreastus}" \
  "${USCT_SAMPLE_ROOT:-$HOME/usct-benchlab/data/openbreastus_sample}" \
  "${USCT_RUN_ROOT:-$HOME/usct-benchlab/runs/usctbench_runs}"; do
  echo "PATH $path"
  ls -lah "$path" 2>/dev/null | head || true
done
