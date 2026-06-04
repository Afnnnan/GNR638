# GNR638 — Deep Learning for Computer Vision

Coursework repository for **GNR 638: Deep Learning for Computer Vision** at IIT Bombay.

**Author:** Afnan Abdul Gafoor (22B2505)

---

## Repository Structure

```
GNR638/
├── Assignment1/   Custom DL Framework (Python + C++)
├── Assignment2/   Pre-trained CNN Backbone Analysis
├── Assignment3/   U-Net for Biomedical Image Segmentation
└── Project/       Visual MCQ Answering with VLMs
```

---

## Assignment 1 — Custom Deep Learning Framework

> Build a CNN training pipeline from scratch — no PyTorch, no TensorFlow.

- **Stack:** Python 3.12 frontend + C++17 backend (loaded via `ctypes`)
- **Features:** Custom tensor ops, conv/relu/maxpool/linear layers (forward + backward), SGD optimizer, checkpoint saving
- **Datasets:** Two image classification datasets (`data_1`, `data_2`)
- **Report:** [`Assignment1/reports/final_report.pdf`](Assignment1/reports/final_report.pdf)

📂 [View Assignment 1](Assignment1/)

---

## Assignment 2 — Pre-trained CNN Backbone Analysis

> Analyze ResNet50, DenseNet121, and EfficientNet-B0 across five experimental scenarios on the Aerial Image Dataset (AID).

| Scenario | Description |
|----------|-------------|
| Linear Probe | Frozen backbone + linear classifier |
| Fine-Tuning | Linear / last-block / full / selective (20%) |
| Few-Shot | 100% / 20% / 5% data regimes |
| Corruption Robustness | Gaussian noise, motion blur, brightness shift |
| Layer Probing | Probes at early / middle / mid-deep / final layers |

**Key finding:** DenseNet121 is the best all-rounder — highest linear probe accuracy, most data-efficient, and most robust to corruption.

- **Report:** [`Assignment2/report.pdf`](Assignment2/report.pdf)

📂 [View Assignment 2](Assignment2/)

---

## Assignment 3 — U-Net: Biomedical Image Segmentation

> Implement the [U-Net paper](https://arxiv.org/abs/1505.04597) (Ronneberger et al., 2015) from scratch and compare against the official Caffe implementation.

Trained and evaluated on all three datasets from the original paper:

| Dataset | Dice | IoU | Pixel Acc |
|---------|------|-----|-----------|
| ISBI 2012 EM Segmentation | 0.946 | 0.898 | 0.918 |
| PhC-C2DH-U373 | 0.963 | 0.930 | 0.995 |
| DIC-C2DH-HeLa | 0.959 | 0.922 | 0.957 |

- **Architecture:** 31M parameters, encoder-decoder with skip connections
- **Training:** SGD (momentum=0.99), weighted cross-entropy with border-emphasis weight maps, elastic deformation augmentation
- **Report:** [`Assignment3/report.pdf`](Assignment3/report.pdf)

📂 [View Assignment 3](Assignment3/)

---

## Final Project — Visual MCQ Answering

> Zero-shot visual question answering on MCQ images using a Vision-Language Model.

- **Model:** Qwen3-VL-8B-Instruct with 4-bit NF4 quantization (BitsAndBytes)
- **Approach:** The model reads MCQ images, reasons through the question, and predicts the correct option — no task-specific training required
- **Target:** Linux, L40s GPU, CUDA 12.6

📂 [View Project](Project/)

---

## Tech Stack

| Component | Technologies |
|-----------|-------------|
| Assignment 1 | Python, C++17, OpenCV, ctypes |
| Assignment 2 | PyTorch, timm, torchvision, scikit-learn |
| Assignment 3 | PyTorch, tifffile, scipy, scikit-image |
| Project | PyTorch, Transformers, BitsAndBytes, Qwen3-VL |
