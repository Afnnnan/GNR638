#!/bin/bash
# =============================================================================
# GNR638 Final Project — Setup Script
# Author: Afnan Abdul Gafoor (22b2505)
# Project: 2 (Deep Learning MCQ Solver)
#
# This script:
#   1. Clones the GitHub repository containing inference.py
#   2. Creates a conda environment (gnr_project_env, Python 3.11)
#   3. Installs PyTorch with CUDA support and all dependencies
#   4. Pre-downloads model weights (Qwen3-VL-8B-Instruct) from HuggingFace
#
# Target: Linux with L40s GPU, CUDA 12.6, 48GB VRAM, 16GB RAM
# =============================================================================

set -e  # Exit on any error

# ── Initialize conda (needed in non-interactive shells / subshells) ──────────
# Try common conda install locations
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
elif command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
else
    echo "ERROR: conda not found!" && exit 1
fi
echo "conda initialized: $(conda --version)"

echo "=============================================="
echo "GNR638 Project 2 — Setup (22b2505)"
echo "=============================================="

# ── Step 1: Clone repository ─────────────────────────────────────────────────
echo ""
echo "[1/4] Cloning repository ..."
if [ -d "repo" ]; then
    echo "  → repo/ already exists, pulling latest ..."
    cd repo && git pull && cd ..
else
    git clone https://github.com/Afnnnan/GNR638-Final-Project.git repo
fi
cp repo/inference.py .
echo "  → inference.py copied to working directory"

# ── Step 2: Create conda environment ─────────────────────────────────────────
echo ""
echo "[2/4] Creating conda environment: gnr_project_env (Python 3.11) ..."

# Remove existing environment if present (clean slate)
conda remove --name gnr_project_env --all -y 2>/dev/null || true

conda create -n gnr_project_env python=3.11 -y

# Activate environment
conda activate gnr_project_env

echo "  → Environment activated: $(python --version)"

# ── Step 3: Install dependencies ─────────────────────────────────────────────
echo ""
echo "[3/4] Installing PyTorch and dependencies ..."

# PyTorch with CUDA 12.6 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# ML dependencies (pinned to match Kaggle environment)
pip install \
    "transformers>=4.52.0" \
    "accelerate>=0.34.0" \
    "bitsandbytes>=0.43.0" \
    "qwen-vl-utils>=0.0.8" \
    pandas

echo "  → All dependencies installed"

# ── Step 4: Pre-download model weights ───────────────────────────────────────
echo ""
echo "[4/4] Pre-downloading Qwen3-VL-8B-Instruct model weights ..."
echo "  (This may take several minutes depending on network speed)"

python -c "
from huggingface_hub import snapshot_download

print('Downloading model weights ...')
snapshot_download('Qwen/Qwen3-VL-8B-Instruct')

print('Downloading processor ...')
# Processor files are included in the model repo, already downloaded above

print('Model weights cached successfully.')
"

echo ""
echo "=============================================="
echo "Setup complete!"
echo "  Environment : gnr_project_env"
echo "  Python      : $(python --version)"
echo "  Inference   : inference.py"
echo "=============================================="
