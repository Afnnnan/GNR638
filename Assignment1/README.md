# Custom Deep Learning Framework (Python Frontend + C++ Backend)

This repository implements a custom CNN training/evaluation pipeline without external deep-learning frameworks.

## Stack
- Frontend: Python 3.12
- Backend: C++17 shared library loaded through `ctypes`
- Image loading: OpenCV (`cv2`) only

### Quick setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install opencv-python
python -c "import cv2; print(cv2.__version__)"
```

## Dataset Requirement
Ensure both dataset directories exist before running:
- `datasets/data_1`
- `datasets/data_2`

Quick check:
```bash
test -d datasets/data_1 && echo "data_1 found" || echo "data_1 missing"
test -d datasets/data_2 && echo "data_2 found" || echo "data_2 missing"
```

Expected structure:
```text
datasets/
  data_1/
    <class_name_1>/*.png
    <class_name_2>/*.png
    ...
  data_2/
    <class_name_1>/*.png
    <class_name_2>/*.png
    ...
```

## Folder Layout
- `cpp_backend/kernels.cpp`: C++ compute kernels (conv, relu, maxpool, linear + backward)
- `dlframework/`: Python framework (tensor, layers, loss, optimizer, model builder, complexity metrics)
- `utils/`: dataset loading/batching, config, checkpoint helpers
- `train.py`: training entrypoint
- `evaluate.py`: evaluation entrypoint
- `configs/`: experiment configs
- `reports/final_report.pdf`: final assignment report

## Build
```bash
python3 build_backend.py
```

## Train
```bash
python3 train.py --dataset_path datasets/data_1 --config_path configs/config_data1_fresh_tuned.json
python3 train.py --dataset_path datasets/data_2 --config_path configs/config_data2_cnn_best_fresh.json
```

## Evaluate
```bash
python3 evaluate.py --dataset_path datasets/data_1 --weights_path outputs/data1_fresh_tuned/model_checkpoint_best.json --out_path outputs/data1_fresh_tuned/eval_metrics.json
python3 evaluate.py --dataset_path datasets/data_2 --weights_path outputs/data2_cnn_best_fresh/model_checkpoint_best.json --out_path outputs/data2_cnn_best_fresh/eval_metrics.json
```

## Resume Training (Optional)
```bash
python3 train.py --dataset_path datasets/data_2 --config_path configs/config_data2_cnn_best_fresh.json --resume_path outputs/data2_cnn_best_fresh/model_checkpoint.json
```

## Notes
- Training and evaluation print dataset loading time, params, MACs, and FLOPs.
- For detailed methodology/results, see [`reports/final_report.pdf`](reports/final_report.pdf).

