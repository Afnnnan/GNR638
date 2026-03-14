"""
Training & evaluation loop, gradient norm tracking, and efficiency metrics
for GNR 638 Assignment 2.
"""
import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

import config


# ─── Efficiency Metrics ─────────────────────────────────────────────────────

def _convert_buffers_to_float32(model):
    """Convert all float64 buffers to float32 (MPS doesn't support float64)."""
    for name, buf in model.named_buffers():
        if buf.dtype == torch.float64:
            # Replace buffer in-place by finding the parent module
            parts = name.split(".")
            mod = model
            for p in parts[:-1]:
                mod = getattr(mod, p)
            mod.register_buffer(parts[-1], buf.float())
    return model


def compute_flops_params(model, input_size=(1, 3, 224, 224), device=None):
    """Compute FLOPs, MACs, and parameter count using thop. Runs on CPU to avoid MPS issues."""
    from thop import profile, clever_format
    import copy
    # Always compute on CPU to avoid MPS float64 issues
    model_copy = copy.deepcopy(model).cpu().float()
    model_copy.eval()
    dummy = torch.randn(*input_size).cpu().float()
    macs, params = profile(model_copy, inputs=(dummy,), verbose=False)
    flops = 2 * macs  # FLOPs ≈ 2 × MACs
    macs_str, params_str = clever_format([macs, params], "%.2f")
    flops_str = clever_format([flops], "%.2f")
    info = {
        "MACs": macs, "MACs_str": macs_str,
        "FLOPs": flops, "FLOPs_str": flops_str,
        "Params": params, "Params_str": params_str,
    }
    print(f"  [Efficiency] MACs: {macs_str} | FLOPs: {flops_str} | Params: {params_str}")
    del model_copy
    return info


# ─── Gradient Norm Tracking ─────────────────────────────────────────────────

def compute_gradient_norms(model):
    """Compute L2 norm of gradients for each parameter group (by layer name prefix)."""
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norm = param.grad.data.norm(2).item()
            # Group by top-level module
            prefix = name.split(".")[0]
            if prefix not in grad_norms:
                grad_norms[prefix] = []
            grad_norms[prefix].append(norm)

    # Average norms per group
    avg_norms = {k: np.mean(v) for k, v in grad_norms.items()}
    return avg_norms


# ─── Training Loop ──────────────────────────────────────────────────────────

def train_model(
    model,
    train_loader,
    val_loader,
    epochs: int = config.EPOCHS_FULL,
    lr: float = config.LR,
    weight_decay: float = config.WEIGHT_DECAY,
    device=None,
    save_path: str = None,
    track_grad_norms: bool = False,
):
    """
    Train a model and return training history.

    Returns:
        history (dict): {
            'train_loss', 'train_acc', 'val_loss', 'val_acc',
            'grad_norms' (optional), 'best_val_acc', 'epoch_times'
        }
    """
    device = device or config.DEVICE
    _convert_buffers_to_float32(model)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Only optimize parameters that require gradients
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(params_to_optimize, lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "grad_norms": [],
        "epoch_times": [],
    }
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Train ──
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.1f}%")

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        # Track gradient norms
        if track_grad_norms:
            grad_norms = compute_gradient_norms(model)
            history["grad_norms"].append(grad_norms)

        # ── Validate ──
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step()
        elapsed = time.time() - t0
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_times"].append(elapsed)

        print(f"  Epoch {epoch}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}% | "
              f"Time: {elapsed:.1f}s")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"  ✓ Saved best model (val_acc={val_acc:.1f}%) → {save_path}")

    history["best_val_acc"] = best_val_acc
    print(f"  [Done] Best val accuracy: {best_val_acc:.1f}%")
    return history


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate(model, loader, criterion=None, device=None):
    """Evaluate model on a data loader. Returns (loss, accuracy)."""
    device = device or config.DEVICE
    criterion = criterion or nn.CrossEntropyLoss()
    model = model.to(device)
    model.eval()

    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total if total > 0 else 0
    accuracy = 100.0 * correct / total if total > 0 else 0
    return avg_loss, accuracy


def get_predictions(model, loader, device=None):
    """Get all predictions and true labels from a loader."""
    device = device or config.DEVICE
    model = model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.append(preds.cpu())
            all_labels.append(labels)

    return torch.cat(all_preds), torch.cat(all_labels)


def extract_features(model, loader, device=None):
    """
    Extract penultimate-layer features and labels from a loader.
    Uses the model's forward_features method (timm models).
    """
    device = device or config.DEVICE
    model = model.to(device)
    model.eval()

    all_features, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            feats = model.forward_features(images)
            # Global average pool if spatial
            if feats.dim() == 4:
                feats = torch.nn.functional.adaptive_avg_pool2d(feats, 1).flatten(1)
            elif feats.dim() == 3:
                feats = feats.mean(dim=1)
            all_features.append(feats.cpu())
            all_labels.append(labels)

    return torch.cat(all_features), torch.cat(all_labels)
