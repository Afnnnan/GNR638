#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import time

from dlframework import CppBackend, SGD, build_model, softmax_cross_entropy_with_logits
from dlframework.metrics import cnn_complexity, resnet_small_complexity
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.config import load_json, save_json
from utils.data import ImageFolderDataset, iter_batches, split_train_val


def argmax_row(data, offset, width):
    best_idx = 0
    best = data[offset]
    for j in range(1, width):
        v = data[offset + j]
        if v > best:
            best = v
            best_idx = j
    return best_idx


def evaluate(model, dataset, samples, batch_size, seed, normalize_mean, normalize_std):
    total = 0
    correct = 0
    total_loss = 0.0

    for x, y in iter_batches(
        dataset,
        samples,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        seed=seed,
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

    return total_loss / max(1, total), correct / max(1, total)


def main():
    parser = argparse.ArgumentParser(description="Train custom CNN with Python frontend + C++ backend")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset parent directory")
    parser.add_argument("--config_path", type=str, required=True, help="Path to JSON config")
    parser.add_argument("--resume_path", type=str, default=None, help="Optional checkpoint path to resume from")
    args = parser.parse_args()

    cfg = load_json(args.config_path)
    seed = int(cfg.get("seed", 42))
    random.seed(seed)

    epochs = int(cfg.get("epochs", 8))
    batch_size = int(cfg.get("batch_size", 64))
    lr = float(cfg.get("lr", 0.01))
    lr_decay_every = int(cfg.get("lr_decay_every", 0))
    lr_decay_gamma = float(cfg.get("lr_decay_gamma", 1.0))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    momentum = float(cfg.get("momentum", 0.0))
    val_ratio = float(cfg.get("val_ratio", 0.1))
    augment = bool(cfg.get("augment", False))
    max_train_samples = cfg.get("max_train_samples")
    max_val_samples = cfg.get("max_val_samples")
    in_channels = int(cfg.get("in_channels", 3))
    image_size = int(cfg.get("image_size", 32))
    model_type = str(cfg.get("model_type", "cnn")).lower()
    conv1_out = int(cfg.get("conv1_out", 8))
    conv2_out = int(cfg.get("conv2_out", 16))
    hidden = int(cfg.get("hidden", 64))
    resnet_base_channels = int(cfg.get("resnet_base_channels", 16))
    out_dir = cfg.get("out_dir", "outputs")
    resume_path = args.resume_path or cfg.get("resume_path")
    normalize_mean = list(cfg.get("normalize_mean", [0.0, 0.0, 0.0]))
    normalize_std = list(cfg.get("normalize_std", [1.0, 1.0, 1.0]))
    augment_crop_pad = int(cfg.get("augment_crop_pad", 0))

    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading dataset index from: {args.dataset_path}")
    dataset = ImageFolderDataset(args.dataset_path, image_size=image_size)
    decode_time_sec = dataset.measure_decode_time()
    print(f"Dataset indexing time (s): {dataset.index.indexing_time_sec:.4f}")
    print(f"Dataset loading/decode time (s): {decode_time_sec:.4f}")

    train_samples, val_samples = split_train_val(dataset.samples, val_ratio=val_ratio, seed=seed)
    if max_train_samples is not None:
        train_samples = train_samples[: int(max_train_samples)]
    if max_val_samples is not None:
        val_samples = val_samples[: int(max_val_samples)]
    num_classes = dataset.num_classes

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
    optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

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

    history = []
    start_epoch = 0
    best_val_acc = -1.0
    best_val_loss = 1e30
    if resume_path:
        print(f"Resuming from checkpoint: {resume_path}")
        loaded_meta = load_checkpoint(resume_path, model)
        start_epoch = int(loaded_meta.get("last_epoch", 0))
        best_val_acc = float(loaded_meta.get("best_val_acc", -1.0))
        best_val_loss = float(loaded_meta.get("best_val_loss", 1e30))

        existing_log_path = os.path.join(out_dir, "train_log.json")
        if os.path.exists(existing_log_path):
            prev_log = load_json(existing_log_path)
            history = list(prev_log.get("history", []))
            if history and start_epoch == 0:
                start_epoch = int(history[-1].get("epoch", 0))
            if history and best_val_acc < 0.0:
                best_val_acc = max(float(r["val_acc"]) for r in history)
            if history:
                best_val_loss = min(float(r["val_loss"]) for r in history)
        print(f"Resume start_epoch={start_epoch} best_val_acc={best_val_acc:.4f}")

    if start_epoch >= epochs:
        print(f"No training needed: start_epoch={start_epoch} >= epochs={epochs}")
        return

    ckpt_path = os.path.join(out_dir, "model_checkpoint.json")
    best_ckpt_path = os.path.join(out_dir, "model_checkpoint_best.json")
    for epoch in range(start_epoch + 1, epochs + 1):
        t0 = time.time()
        if lr_decay_every > 0:
            decay_steps = (epoch - 1) // lr_decay_every
            optimizer.lr = lr * (lr_decay_gamma ** decay_steps)

        optimizer.zero_grad()

        total = 0
        correct = 0
        running_loss = 0.0

        for x, y in iter_batches(
            dataset,
            train_samples,
            batch_size=batch_size,
            shuffle=True,
            augment=augment,
            seed=seed + epoch,
            augment_crop_pad=augment_crop_pad,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        ):
            logits = model.forward(x)
            loss, grad_logits = softmax_cross_entropy_with_logits(logits, y)

            model.backward(grad_logits)
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss * len(y)

            n, c = logits.shape
            for i in range(n):
                pred = argmax_row(logits.data, i * c, c)
                if pred == y[i]:
                    correct += 1
            total += len(y)

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)
        val_loss, val_acc = evaluate(
            model,
            dataset,
            val_samples,
            batch_size=batch_size,
            seed=seed,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )
        epoch_time = time.time() - t0

        rec = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch_time_sec": epoch_time,
            "lr": optimizer.lr,
        }
        history.append(rec)

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={optimizer.lr:.6f} epoch_time_sec={epoch_time:.2f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            metadata_best = {
                "num_classes": num_classes,
                "class_names": [dataset.index.label_to_name[i] for i in range(num_classes)],
                "config": cfg,
                "model_type": model_type,
                "dataset_indexing_time_sec": dataset.index.indexing_time_sec,
                "dataset_decode_time_sec": decode_time_sec,
                "complexity": complexity,
                "last_epoch": epoch,
                "best_val_acc": best_val_acc,
                "best_val_loss": best_val_loss,
            }
            save_checkpoint(best_ckpt_path, model, metadata_best)

        metadata = {
            "num_classes": num_classes,
            "class_names": [dataset.index.label_to_name[i] for i in range(num_classes)],
            "config": cfg,
            "model_type": model_type,
            "dataset_indexing_time_sec": dataset.index.indexing_time_sec,
            "dataset_decode_time_sec": decode_time_sec,
            "complexity": complexity,
            "last_epoch": epoch,
            "best_val_acc": best_val_acc,
            "best_val_loss": best_val_loss,
        }
        save_checkpoint(ckpt_path, model, metadata)
        save_json(os.path.join(out_dir, "train_log.json"), {"history": history, "metadata": metadata})

    print(f"Saved checkpoint: {ckpt_path}")
    print(f"Saved best checkpoint: {best_ckpt_path}")
    print(f"Saved training log: {os.path.join(out_dir, 'train_log.json')}")


if __name__ == "__main__":
    main()
