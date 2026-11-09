#!/usr/bin/env bash
# Environment + data setup for 11-711 F2026 Homework 1.
set -euo pipefail

ENV_NAME="${ENV_NAME:-anlp-hw1}"
PY_VERSION="3.11"

# ---- data source ---------------------------------------------------------
# Data is hosted on the HuggingFace Hub (resumable downloads, content hashing).
# Set HF_DATA_REPO to fetch from a mirror.
HF_DATA_REPO="${HF_DATA_REPO:-aashiqmuhamed/anlp-f2026-hw1}"
HF_REVISION="${HF_REVISION:-main}"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

# ---- conda environment ---------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html" >&2
  exit 1
fi

if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
  conda create -y -n "$ENV_NAME" "python=${PY_VERSION}"
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Install the CUDA build on Linux, the default (MPS/CPU) build elsewhere.
if [[ "$(uname -s)" == "Linux" ]]; then
  pip install torch --index-url https://download.pytorch.org/whl/cu121
else
  pip install torch
fi

pip install -r requirements.txt
pip install -e .

# ---- data ----------------------------------------------------------------
mkdir -p data

echo "Fetching data files from ${HF_DATA_REPO}..."
python - <<PYEOF
import shutil, sys
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO = "${HF_DATA_REPO}"
REV = "${HF_REVISION}"
FILES = ["stories42M.pt", "tinystories-tokens.npz"]

for name in FILES:
    dest = Path("data") / name
    if dest.exists():
        print(f"  {dest} already present")
        continue
    try:
        cached = hf_hub_download(repo_id=REPO, filename=name,
                                 repo_type="dataset", revision=REV)
    except Exception as exc:
        print(f"  FAILED to fetch {name}: {exc}", file=sys.stderr)
        print(f"  If the download keeps failing, ask on Piazza.", file=sys.stderr)
        sys.exit(1)
    # copy out of the shared cache so the repo is self-contained
    shutil.copy(cached, dest)
    print(f"  {dest} ({dest.stat().st_size/1e6:.0f}MB)")
PYEOF

echo
echo "Setup complete. Verify with:  pytest tests/ -q"
echo "Activate later with:          conda activate ${ENV_NAME}"
