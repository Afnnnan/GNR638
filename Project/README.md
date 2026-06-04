# GNR638 Final Project — Deep Learning MCQ Solver

**Author:** Afnan Abdul Gafoor (22b2505)  
**Project:** 2 — Visual MCQ Answering

## Approach

Zero-shot visual question answering using **Qwen3-VL-8B-Instruct** with 4-bit NF4 quantization via BitsAndBytes. The model reads MCQ images, reasons through the question, and predicts the correct option — no task-specific training required.

## Setup & Usage

```bash
bash setup.bash
conda activate gnr_project_env
python inference.py --test_dir <absolute_path_to_test_dir>
```

### What `setup.bash` does:
1. Clones this repository
2. Creates conda environment (`gnr_project_env`, Python 3.11)
3. Installs PyTorch (CUDA 12.6), transformers, accelerate, bitsandbytes, qwen-vl-utils, pandas
4. Pre-downloads Qwen3-VL-8B-Instruct model weights from HuggingFace

### Output
`submission.csv` with columns: `id`, `image_name`, `option` (1-4 = A-D, 5 = skip)

## Target Environment

- Linux, L40s GPU, CUDA 12.6, 48GB VRAM, 16GB RAM
