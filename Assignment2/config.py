"""
Global configuration for GNR 638 Assignment 2.
Device-agnostic: CUDA -> MPS -> CPU fallback.
"""
import os
import random
import torch
import numpy as np

# ─── Seed ────────────────────────────────────────────────────────────────────
SEED = 42

def set_seed(seed: int = SEED):
    """Set seed for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ─── Device ──────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()

# ─── Paths ───────────────────────────────────────────────────────────────────
# Adjust DATA_ROOT if running on Colab/Kaggle
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "Assignment 2 Datasets", "train_data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ─── Model Config ────────────────────────────────────────────────────────────
MODEL_NAMES = ["resnet50", "densenet121", "efficientnet_b0"]
NUM_CLASSES = 30

# ─── Training Hyperparams ───────────────────────────────────────────────────
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS_FULL = 30          # full-data training
EPOCHS_FEWSHOT = 20       # few-shot settings
NUM_WORKERS = 2           # DataLoader workers

# ─── Image Config ────────────────────────────────────────────────────────────
IMG_SIZE = 224            # 299 for InceptionV3, handled in dataset.py
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ─── Few-Shot Fractions ─────────────────────────────────────────────────────
FEWSHOT_FRACTIONS = [1.0, 0.2, 0.05]

# ─── Corruption Config ──────────────────────────────────────────────────────
GAUSSIAN_NOISE_SIGMAS = [0.05, 0.1, 0.2]

# ─── Train/Val Split ────────────────────────────────────────────────────────
VAL_FRACTION = 0.2

_PRINTED = False
if not _PRINTED:
    print(f"[Config] Device: {DEVICE} | Seed: {SEED} | Data root: {DATA_ROOT}")
    _PRINTED = True
