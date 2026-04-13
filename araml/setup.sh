#!/usr/bin/env bash
# setup.sh — Install ARAML dependencies on a GPU machine (Linux/WSL).
#
# Usage:
#   chmod +x setup.sh && ./setup.sh
#
# Tested with: Python 3.10/3.11, CUDA 11.8 / 12.1, RTX 4000

set -e

PYTHON=${PYTHON:-python3}

# ── 1. Create virtual environment ────────────────────────────────────────────
if [ ! -d "venv" ]; then
    echo "[1/5] Creating virtual environment..."
    $PYTHON -m venv venv
else
    echo "[1/5] Virtual environment already exists, skipping."
fi

source venv/bin/activate
pip install --upgrade pip --quiet

# ── 2. Install PyTorch with CUDA 12.1 ────────────────────────────────────────
# Change cu121 → cu118 if your driver supports CUDA 11.8 only.
# Check with: nvidia-smi
echo "[2/5] Installing PyTorch (CUDA 12.1)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet

# ── 3. Install project requirements ──────────────────────────────────────────
echo "[3/5] Installing project requirements..."
pip install -r requirements.txt --quiet

# ── 4. Install faiss-gpu (replaces faiss-cpu for CUDA acceleration) ──────────
echo "[4/5] Installing faiss-gpu..."
pip install faiss-gpu --quiet 2>/dev/null || {
    echo "  faiss-gpu not available via pip — trying conda-forge..."
    if command -v conda &>/dev/null; then
        conda install -c conda-forge faiss-gpu -y --quiet
    else
        echo "  WARNING: faiss-gpu install failed. faiss-cpu (already installed) will be used."
    fi
}

# ── 5. Verify ─────────────────────────────────────────────────────────────────
echo "[5/5] Verifying installation..."
$PYTHON - <<'EOF'
import torch, transformers, datasets, faiss, numpy, yaml, tqdm
print(f"  torch       : {torch.__version__}  (CUDA available: {torch.cuda.is_available()})")
if torch.cuda.is_available():
    print(f"  GPU         : {torch.cuda.get_device_name(0)}")
print(f"  transformers: {transformers.__version__}")
print(f"  datasets    : {datasets.__version__}")
print(f"  faiss       : {faiss.__version__}")
print(f"  numpy       : {numpy.__version__}")
print("All dependencies OK.")
EOF

echo ""
echo "Setup complete. Activate with:  source venv/bin/activate"
echo "Then run:  cd araml && PYTHONPATH=. python scripts/train.py --epochs 20"
