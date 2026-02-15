#!/usr/bin/env python3
from __future__ import annotations

import os
import argparse

from dlframework import CppBackend, build_model, softmax_cross_entropy_with_logits
from dlframework.metrics import cnn_complexity, resnet_small_complexity
from utils.checkpoint import load_checkpoint
from utils.config import load_json, save_json
from utils.data import ImageFolderDataset, iter_batches


def argmax_row(data, offset, width):
    best_idx = 0
    best = data[offset]
    for j in range(1, width):
        v = data[offset + j]
        if v > best:
            best = v
            best_idx = j
    return best_idx


def main():
    parser = argparse.ArgumentParser(description="Evaluate custom CNN")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to test dataset parent directory")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to saved checkpoint JSON")
    parser.add_argument("--out_path", type=str, default="outputs/eval_metrics.json", help="Where to save metrics JSON")
    args = parser.parse_args()

    dataset = ImageFolderDataset(args.dataset_path, image_size=32)
    decode_time_sec = dataset.measure_decode_time()
    print(f"Dataset indexing time (s): {dataset.index.indexing_time_sec:.4f}")
    print(f"Dataset loading/decode time (s): {decode_time_sec:.4f}")

    state = load_json(args.weights_path)
    metadata = state.get("metadata", {})
    cfg = metadata.get("config", {})
    num_classes = int(metadata.get("num_classes", dataset.num_classes))
    in_channels = int(cfg.get("in_channels", 3))
    image_size = int(cfg.get("image_size", 32))
    model_type = str(metadata.get("model_type", cfg.get("model_type", "cnn"))).lower()
    conv1_out = int(cfg.get("conv1_out", 8))
    conv2_out = int(cfg.get("conv2_out", 16))
    hidden = int(cfg.get("hidden", 64))
    resnet_base_channels = int(cfg.get("resnet_base_channels", 16))

    backend = CppBackend(cfg.get("backend_lib_path"))
    model = build_model(
        backend,
        num_classes=num_classes,
        model_type=model_type,
        in_channels=in_channels,
        in_size=image_size,
        conv1_out=conv1_out,
        conv2_out=conv2_out,
        hidden=hidden,
        resnet_base_channels=resnet_base_channels,
    )
    metadata = load_checkpoint(args.weights_path, model)

    if model_type == "resnet":
        complexity = resnet_small_complexity(
            in_channels=in_channels,
            in_h=image_size,
            in_w=image_size,
            num_classes=num_classes,
            base_channels=resnet_base_channels,
        )
    else:
        complexity = cnn_complexity(
            in_channels=in_channels,
            in_h=image_size,
            in_w=image_size,
            num_classes=num_classes,
            conv1_out=conv1_out,
            conv2_out=conv2_out,
            hidden=hidden,
        )
    print(f"Trainable parameters: {complexity['params']}")
    print(f"MACs per forward pass (single sample): {complexity['macs']}")
    print(f"FLOPs per forward pass (single sample): {complexity['flops']}")

    total = 0
    correct = 0
    total_loss = 0.0
    batch_size = int(cfg.get("batch_size", 64))

    normalize_mean = list(cfg.get("normalize_mean", [0.0, 0.0, 0.0]))
    normalize_std = list(cfg.get("normalize_std", [1.0, 1.0, 1.0]))

    for x, y in iter_batches(
        dataset,
        dataset.samples,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        seed=0,
        augment_crop_pad=0,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    ):
        logits = model.forward(x)
        loss, _ = softmax_cross_entropy_with_logits(logits, y)
        total_loss += loss * len(y)

        n, c = logits.shape
        for i in range(n):
            pred = argmax_row(logits.data, i * c, c)
            if pred == y[i]:
                correct += 1
        total += len(y)

    metrics = {
        "loss": total_loss / max(1, total),
        "accuracy": correct / max(1, total),
        "num_samples": total,
        "dataset_indexing_time_sec": dataset.index.indexing_time_sec,
        "dataset_decode_time_sec": decode_time_sec,
        "complexity": complexity,
        "checkpoint_metadata": metadata,
    }

    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_json(args.out_path, metrics)

    print(f"Evaluation loss: {metrics['loss']:.4f}")
    print(f"Evaluation accuracy: {metrics['accuracy']:.4f}")
    print(f"Saved metrics: {args.out_path}")


if __name__ == "__main__":
    main()
