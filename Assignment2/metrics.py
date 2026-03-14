"""
Metrics computation and reporting for GNR 638 Assignment 2.
Confusion matrices, accuracy tables, FLOPs/params reporting.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import config


def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix",
                          save_path=None):
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved confusion matrix → {save_path}")
    plt.close()
    return cm


def plot_accuracy_curves(history, title="Training Curves", save_path=None):
    """Plot train/val accuracy and loss curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{title} — Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-", label="Train Acc")
    ax2.plot(epochs, history["val_acc"], "r-", label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"{title} — Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved accuracy curves → {save_path}")
    plt.close()


def plot_convergence_comparison(histories_dict, title="Convergence Comparison",
                                 save_path=None):
    """
    Plot training loss vs epoch for multiple strategies/models.
    histories_dict: {label: history, ...}
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, hist in histories_dict.items():
        epochs = range(1, len(hist["train_loss"]) + 1)
        ax.plot(epochs, hist["train_loss"], label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved convergence plot → {save_path}")
    plt.close()


def print_classification_report(y_true, y_pred, class_names):
    """Print a formatted classification report."""
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))


def create_results_table(data, columns, title="Results", save_path=None):
    """
    Create and save a results table as an image.
    data: list of lists (rows)
    columns: list of column headers
    """
    import pandas as pd
    df = pd.DataFrame(data, columns=columns)
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    print(f"{'='*60}\n")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Save as CSV
        csv_path = save_path.replace(".png", ".csv")
        df.to_csv(csv_path, index=False)
        print(f"  ✓ Saved table → {csv_path}")

        # Also save as image
        fig, ax = plt.subplots(figsize=(max(12, len(columns) * 2.5), max(3, len(data) * 0.5 + 1.5)))
        ax.axis("off")
        table = ax.table(cellText=df.values, colLabels=df.columns,
                         cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved table image → {save_path}")

    return df
