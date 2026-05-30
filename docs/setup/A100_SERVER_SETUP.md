# A100 Server Setup Guide for `usct-benchlab`

This guide describes what the user should prepare on the A100 server before asking Codex to build the library. The goal is to remove avoidable friction: GitHub SSH, CUDA/PyTorch environment, data paths, external-code paths, and run directories.

## 1. Verify GPU and SSH access

```bash
nvidia-smi
ssh -T git@github.com
```

Expected GitHub result is something like:

```text
Hi Math-Wu/usct-benchlab! You've successfully authenticated...
```

If GitHub SSH fails, fix this before Codex starts. Codex will waste time if it cannot clone/push.

## 2. Clone the private repository

```bash
mkdir -p ~/projects
cd ~/projects
git clone git@github.com:Math-Wu/usct-benchlab.git
cd usct-benchlab
git config user.name "Math-Wu"
git config user.email "wdl200268@gmail.com"
```

If the repo is empty, Codex should initialize the package skeleton.

## 3. Prepare data and run directories

Recommended layout:

```bash
sudo mkdir -p /data/openbreastus
sudo mkdir -p /data/openbreastus_sample
sudo mkdir -p /data/usctbench_runs
sudo chown -R $USER:$USER /data/openbreastus /data/openbreastus_sample /data/usctbench_runs
```

Then put the OpenBreastUS dataset under:

```text
/data/openbreastus
```

Create a small note for Codex:

```bash
cat > /data/openbreastus/README_LOCAL_LAYOUT.txt <<'EOF'
This directory contains the locally downloaded OpenBreastUS dataset.
Do not download the full dataset again.
Inspect this directory and write an index before assuming paths.
EOF
```

Inside the repo:

```bash
mkdir -p data/raw data/processed data/interim runs checkpoints external third_party
ln -sfn /data/openbreastus data/raw/openbreastus
ln -sfn /data/openbreastus_sample data/processed/openbreastus_sample
ln -sfn /data/usctbench_runs runs/a100
```

## 4. Environment variables

Create `.env` locally but do not commit it:

```bash
cat > .env <<'EOF'
USCT_DATA_ROOT=/data/openbreastus
USCT_SAMPLE_ROOT=/data/openbreastus_sample
USCT_RUN_ROOT=/data/usctbench_runs
CUDA_VISIBLE_DEVICES=0
OMP_NUM_THREADS=8
MKL_NUM_THREADS=8
EOF
```

Also create `.env.example` in git without private paths if it does not exist.

## 5. Python environment

A100 usually works well with a modern CUDA PyTorch build. Prefer Python 3.10 or 3.11.

### Option A: mamba/conda

```bash
mamba create -n usctbench python=3.10 -y
mamba activate usctbench
python -m pip install -U pip wheel setuptools
```

Install core dependencies:

```bash
pip install numpy scipy h5py zarr pydantic pyyaml typer rich tqdm matplotlib scikit-image pandas xarray
pip install pytest pytest-cov ruff mypy pre-commit
```

Install PyTorch after checking the CUDA driver:

```bash
nvidia-smi
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

If CUDA 12.1 wheel fails, use the official PyTorch selector and install the matching wheel. Do not spend time fighting CUDA before the CPU unit tests pass.

### Optional packages

Install these only when needed:

```bash
pip install astra-toolbox odl deepwave k-wave-python tensorboard wandb
```

`astra-toolbox` can accelerate tomography primitives. `deepwave` can support differentiable wave propagation/FWI. `k-wave-python` is useful for acoustic simulations but may require extra setup.

## 6. MATLAB availability

Some classic USCT code is MATLAB-first. Check whether MATLAB exists:

```bash
which matlab || true
matlab -batch "disp(version)" || true
```

If MATLAB is not available, Codex should implement adapters that skip gracefully. Do not block v0.1 on MATLAB.

Optional environment variable:

```bash
export MATLAB_BIN=$(which matlab)
```

## 7. External code cache

To speed Codex up, optionally pre-clone public reference repositories under `external/`. Check licenses before vendoring or modifying code.

```bash
mkdir -p external
cd external

git clone https://github.com/ucl-bug/ust-sart.git ust-sart || true
git clone https://github.com/rehmanali1994/refractionCorrectedUSCT.github.io.git refractionCorrectedUSCT || true
git clone https://github.com/Ash1362/ray-based-quantitative-ultrasound-tomography.git r-wave || true
git clone https://github.com/rehmanali1994/WaveformInversionUST.git WaveformInversionUST || true
git clone https://github.com/rehmanali1994/FrequencyDifferencing.git FrequencyDifferencing || true
cd ..
```

If the private repo should not contain external source code, keep `external/` ignored and write adapter docs instead.

## 8. Sanity script before Codex starts

```bash
cat > scripts/check_server.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
pwd
hostname
nvidia-smi || true
python --version
python - <<'PY'
import sys
print(sys.executable)
try:
    import torch
    print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
    if torch.cuda.is_available(): print(torch.cuda.get_device_name(0))
except Exception as e:
    print('torch import failed:', repr(e))
PY
for p in "${USCT_DATA_ROOT:-/data/openbreastus}" "${USCT_SAMPLE_ROOT:-/data/openbreastus_sample}" "${USCT_RUN_ROOT:-/data/usctbench_runs}"; do
    echo "PATH $p"
    ls -lah "$p" | head || true
done
git status || true
EOF
chmod +x scripts/check_server.sh
./scripts/check_server.sh
```

## 9. What not to do

- Do not upload OpenBreastUS data to GitHub.
- Do not start with full-resolution FWI.
- Do not assume OpenBreastUS file names; inspect them.
- Do not tune algorithms before checking geometry, units, and masks.
- Do not mix MATLAB output formats with Python output formats; always convert to `ReconstructionResult`.
