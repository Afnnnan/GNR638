# GNR 638: Assignment 2 - Pre-trained CNN Backbone Analysis

**Group 58 | Afnan Abdul Gafoor | 22B2505**  
Indian Institute of Technology Bombay

---

## Overview

This repository analyzes three pre-trained CNN backbones (**ResNet50**, **DenseNet121**, **EfficientNet-B0**) on the [Aerial Image Dataset (AID)](https://arxiv.org/pdf/1608.05167) across five experimental scenarios:

| Scenario | Description |
|----------|-------------|
| 4.1 Linear Probe | Frozen backbone + linear classifier |
| 4.2 Fine-Tuning | 4 strategies: linear / last-block / full / selective (20%) |
| 4.3 Few-Shot | 100% / 20% / 5% data regimes |
| 4.4 Corruption | Gaussian noise, motion blur, brightness shift |
| 4.5 Layer Probing | Linear probes at early / middle / mid-deep / final layers |

---

## Project Structure

```
Assignment2/
├── config.py                  # Hyperparameters, device detection, paths
├── dataset.py                 # Dataset loading, splits, corruption transforms
├── models.py                  # Model factory, freeze strategies, feature hooks
├── train.py                   # Training loop, evaluation, FLOPs/MACs
├── metrics.py                 # Confusion matrix, summary tables
├── visualize.py               # t-SNE, PCA, gradient norms, layer plots
├── run_all.py                 # Master runner (CLI)
├── requirements.txt
├── report.pdf                 # Technical report
└── scenarios/
    ├── scenario_1_linear_probe.py
    ├── scenario_2_finetuning.py
    ├── scenario_3_fewshot.py
    ├── scenario_4_corruption.py
    └── scenario_5_layerwise_probe.py
```

---

## Setup

### Requirements

```bash
pip install torch torchvision timm thop numpy matplotlib seaborn \
            scikit-learn pandas tqdm Pillow scipy
```

Or install all at once:

```bash
pip install -r requirements.txt
```

### Dataset

Download the AID dataset from the [course drive link](https://drive.google.com/drive/folders/1mX8kaByeedwpp9-4-enuJffGChgmdy9n?usp=drive_link) and place it so the structure is:

```
Assignment2/
└── Assignment 2 Datasets/
    └── train_data/
        ├── Airport/
        ├── Beach/
        ├── ...
        └── Viaduct/
```

---

## Running Experiments

### Run all 5 scenarios (full training, 30 epochs)

```bash
python run_all.py
```

### Run specific scenarios

```bash
python run_all.py --scenario 1 2         # Run scenarios 1 and 2
python run_all.py --scenario 4 5         # Run scenarios 4 and 5
```

### Smoke test (2 epochs — pipeline validation only)

```bash
python run_all.py --smoke-test
```

---

## Running on Colab / Kaggle

```python
# Clone the repo
!git clone https://YOUR_TOKEN@github.com/Afnnnan/GNR638.git
%cd GNR638/Assignment2
!pip install timm thop scipy -q

# Set dataset path
import os
os.environ["DATA_ROOT"] = "/kaggle/input/YOUR_DATASET/train_data"  # Kaggle
# os.environ["DATA_ROOT"] = "/content/drive/MyDrive/.../train_data"  # Colab

# Run
!python run_all.py
```

> The `DATA_ROOT` environment variable overrides the default dataset path and can be set without editing any files.

---

## Results Summary

### Scenario 1 — Linear Probe

| Model | Val Acc | FLOPs |
|-------|---------|-------|
| ResNet50 | 83.20% | 8.26G |
| DenseNet121 | **91.92%** | 5.67G |
| EfficientNet-B0 | 84.56% | 769M |

### Scenario 2 — Fine-Tuning (best per model)

| Model | Strategy | Val Acc |
|-------|----------|---------|
| ResNet50 | Last Block | **97.28%** |
| DenseNet121 | Full | 97.07% |
| EfficientNet-B0 | Last Block | **97.43%** |

### Scenario 3 — Few-Shot (Δ drop: 100% → 5%)

| Model | 5% Acc | Δ Drop |
|-------|--------|--------|
| ResNet50 | 79.56% | 18.16% |
| DenseNet121 | 82.13% | **15.39%** |
| EfficientNet-B0 | 75.13% | 22.78% |

### Scenario 4 — Corruption Robustness (Relative Robustness @ σ=0.1)

| Model | Gauss σ=0.1 | Motion Blur | Brightness |
|-------|-------------|-------------|------------|
| ResNet50 | 0.580 | 0.394 | 0.985 |
| DenseNet121 | **0.747** | **0.548** | 0.984 |
| EfficientNet-B0 | 0.358 | 0.461 | **0.991** |

### Scenario 5 — Best Layer for Probing

| Model | Best Layer | Val Acc |
|-------|-----------|---------|
| ResNet50 | Mid-Deep | **93.92%** |
| DenseNet121 | Final | 91.57% |
| EfficientNet-B0 | Final | 89.56% |

---

## Key Findings

- **DenseNet121 is the best all-rounder**: highest linear probe accuracy, most data-efficient, most robust to corruption.
- **Last-block fine-tuning matches full fine-tuning** for all models, offering a more efficient training strategy.
- **ResNet50's mid-deep layer outperforms its final layer** in probing (93.9% vs 88.2%). Final-layer compression hurts transfer.
- **EfficientNet-B0 is most compute-efficient** (10.7x fewer FLOPs than ResNet50) but most fragile under noise and data scarcity.

---

## Reproducibility

- Fixed seed: **42** (applied across Python, NumPy, PyTorch, CUDA)
- Stratified train/val split: 80/20
- Hardware: NVIDIA P100 GPU (Kaggle), total compute: ~8h 51m
- All outputs are deterministic given the same environment and seed

---

## Report

See [`report.pdf`](report.pdf) for the full technical report including all plots, tables, analysis, unexpected findings, and failure cases.
