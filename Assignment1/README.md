# Custom Deep Learning Framework (Python Frontend + C++ Backend)

This repository implements a custom CNN training/evaluation pipeline without external deep-learning frameworks.

## Stack
- Frontend: Python 3.12
- Backend: C++17 shared library loaded through `ctypes`
- Image loading: OpenCV (`cv2`) only

## Folder Layout
- `cpp_backend/kernels.cpp`: C++ compute kernels (conv, relu, maxpool, linear + backward)
- `dlframework/`: Python framework (tensor, layers, loss, optimizer, model builder, complexity metrics)
- `utils/`: dataset loading/batching, config, checkpoint helpers
- `train.py`: training entrypoint
- `evaluate.py`: evaluation entrypoint
- `configs/`: example training configs
- `reports/report_template.md`: report scaffold

## Build
```bash
python3 build_backend.py
```

## Train
```bash
python3 train.py --dataset_path datasets/data_1 --config_path configs/config_data1.json
python3 train.py --dataset_path datasets/data_2 --config_path configs/config_data2.json
```

Train ResNet variant (toggle via config):
```bash
python3 train.py --dataset_path datasets/data_2 --config_path configs/config_data2_resnet.json
```

Tuned CNN continuation for data2 (recommended from existing checkpoint):
```bash
python3 train.py --dataset_path datasets/data_2 --config_path configs/config_data2_cnn_continue_tuned.json
```

Tuned CNN fresh run for data2 (stronger recipe):
```bash
python3 train.py --dataset_path datasets/data_2 --config_path configs/config_data2_cnn_fresh_strong.json
```

Best balanced data2 CNN run (recommended for submission runtime cap):
```bash
python3 train.py --dataset_path datasets/data_2 --config_path configs/config_data2_cnn_best_fresh.json
```

Resume training (example for data2 continuation):
```bash
python3 train.py --dataset_path datasets/data_2 --config_path configs/config_data2_continue.json
```

## Evaluate
```bash
python3 evaluate.py --dataset_path datasets/data_1 --weights_path outputs/data1/model_checkpoint.json --out_path outputs/data1/eval_metrics.json
python3 evaluate.py --dataset_path datasets/data_2 --weights_path outputs/data2/model_checkpoint.json --out_path outputs/data2/eval_metrics.json
```

## Generate Report Plots (SVG, no extra libraries)
```bash
python3 tools/make_plots.py --log_path outputs/data1/train_log.json --out_dir reports/plots/data1
python3 tools/make_plots.py --log_path outputs/data2/train_log.json --out_dir reports/plots/data2
```

## Grader Input Contract
- Training expects only:
  - `--dataset_path <train_dir>`
  - `--config_path <config_json>`
- Evaluation expects only:
  - `--dataset_path <test_dir>`
  - `--weights_path <checkpoint_json>`

No code edits are required between runs.

## Notes
- Ensure OpenCV Python bindings are installed (`cv2` import must work).
- Dataset indexing time and full decode/loading time are printed in both training and evaluation.
- Trainable parameters, MACs, and FLOPs are printed in both training and evaluation.
- Logs are saved in `<out_dir>/train_log.json` and evaluation metrics in `eval_metrics.json`.
