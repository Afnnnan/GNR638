"""
Visualization utilities for GNR 638 Assignment 2.
t-SNE, PCA embeddings, gradient norm plots, layer-wise analysis.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import config


# ─── Embedding Visualizations ───────────────────────────────────────────────

def plot_tsne(features, labels, class_names, title="t-SNE Visualization",
              save_path=None, perplexity=30, n_iter=1000):
    """Plot 2D t-SNE of feature embeddings, colored by class."""
    if isinstance(features, np.ndarray):
        feats = features
    else:
        feats = features.numpy()
    if isinstance(labels, np.ndarray):
        labs = labels
    else:
        labs = labels.numpy()

    print(f"  Computing t-SNE (n={len(feats)})...")
    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(feats)-1),
                n_iter=n_iter, random_state=config.SEED)
    embedded = tsne.fit_transform(feats)

    _plot_2d_embedding(embedded, labs, class_names, title, save_path, method="t-SNE")


def plot_pca(features, labels, class_names, title="PCA Visualization",
             save_path=None):
    """Plot 2D PCA of feature embeddings, colored by class."""
    if isinstance(features, np.ndarray):
        feats = features
    else:
        feats = features.numpy()
    if isinstance(labels, np.ndarray):
        labs = labels
    else:
        labs = labels.numpy()

    pca = PCA(n_components=2, random_state=config.SEED)
    embedded = pca.fit_transform(feats)
    var_explained = pca.explained_variance_ratio_

    full_title = f"{title}\n(PC1: {var_explained[0]:.1%}, PC2: {var_explained[1]:.1%})"
    _plot_2d_embedding(embedded, labs, class_names, full_title, save_path, method="PCA")


def _plot_2d_embedding(embedded, labels, class_names, title, save_path, method=""):
    """Helper: scatter plot of 2D embeddings."""
    n_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = plt.cm.get_cmap("tab20", n_classes)

    for i in range(n_classes):
        mask = labels == i
        if mask.sum() > 0:
            ax.scatter(embedded[mask, 0], embedded[mask, 1],
                       c=[cmap(i)], s=10, alpha=0.6, label=class_names[i])

    ax.set_title(title, fontsize=13)
    ax.set_xlabel(f"{method} Dim 1")
    ax.set_ylabel(f"{method} Dim 2")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=6,
              markerscale=2, ncol=2)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved {method} plot → {save_path}")
    plt.close()


# ─── Gradient Norm Plots ────────────────────────────────────────────────────

def plot_gradient_norms(grad_norms_history, title="Gradient Norms", save_path=None):
    """
    Plot gradient norm bar chart across layer groups.
    grad_norms_history: list of dicts (one per epoch), each: {layer_prefix: avg_norm}
    Uses the last epoch's norms.
    """
    if not grad_norms_history:
        return
    last_norms = grad_norms_history[-1]
    layers = list(last_norms.keys())
    norms = [last_norms[l] for l in layers]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(layers)), norms, color="steelblue", alpha=0.8)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Average Gradient L2 Norm")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved gradient norms → {save_path}")
    plt.close()


def plot_gradient_norms_comparison(all_grad_norms, title="Gradient Norms Comparison",
                                    save_path=None):
    """
    Compare gradient norms across strategies.
    all_grad_norms: {strategy_label: grad_norms_dict}
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    x_labels = set()
    for _, norms in all_grad_norms.items():
        x_labels.update(norms.keys())
    x_labels = sorted(x_labels)

    n_strategies = len(all_grad_norms)
    width = 0.8 / max(n_strategies, 1)

    for i, (label, norms) in enumerate(all_grad_norms.items()):
        vals = [norms.get(l, 0) for l in x_labels]
        x = np.arange(len(x_labels)) + i * width
        ax.bar(x, vals, width=width, label=label, alpha=0.8)

    ax.set_xticks(np.arange(len(x_labels)) + width * (n_strategies - 1) / 2)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Average Gradient L2 Norm")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved gradient norms comparison → {save_path}")
    plt.close()


# ─── Accuracy vs Unfrozen Parameters ────────────────────────────────────────

def plot_accuracy_vs_unfrozen(data, title="Accuracy vs % Unfrozen Parameters",
                               save_path=None):
    """
    data: list of (pct_trainable, val_acc, label)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for pct, acc, label in data:
        ax.scatter(pct, acc, s=100, zorder=5)
        ax.annotate(label, (pct, acc), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)
    ax.set_xlabel("% Trainable Parameters")
    ax.set_ylabel("Best Validation Accuracy (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved accuracy vs unfrozen → {save_path}")
    plt.close()


# ─── Layer-Wise Probing Plots ───────────────────────────────────────────────

def plot_layerwise_accuracy(layer_results, title="Layer-Wise Probing Accuracy",
                             save_path=None):
    """
    layer_results: {model_name: {depth_label: val_acc}}
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, depth_accs in layer_results.items():
        depths = list(depth_accs.keys())
        accs = [depth_accs[d] for d in depths]
        ax.plot(depths, accs, "o-", label=model_name, linewidth=2, markersize=8)

    ax.set_xlabel("Layer Depth")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved layer-wise accuracy → {save_path}")
    plt.close()


def plot_feature_norms(layer_norms, title="Feature Norm Statistics",
                        save_path=None):
    """
    layer_norms: {model_name: {depth_label: (mean_norm, std_norm)}}
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    n_models = len(layer_norms)
    all_depths = set()
    for model_name, depth_dict in layer_norms.items():
        all_depths.update(depth_dict.keys())
    all_depths = sorted(all_depths)
    width = 0.8 / max(n_models, 1)

    for i, (model_name, depth_dict) in enumerate(layer_norms.items()):
        means = [depth_dict.get(d, (0, 0))[0] for d in all_depths]
        stds  = [depth_dict.get(d, (0, 0))[1] for d in all_depths]
        x = np.arange(len(all_depths)) + i * width
        ax.bar(x, means, width=width, yerr=stds, label=model_name,
               alpha=0.8, capsize=3)

    ax.set_xticks(np.arange(len(all_depths)) + width * (n_models - 1) / 2)
    ax.set_xticklabels(all_depths, fontsize=10)
    ax.set_ylabel("Feature L2 Norm")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved feature norms → {save_path}")
    plt.close()


def plot_pca_across_layers(features_dict, labels, class_names,
                            title_prefix="PCA Across Layers", save_path_dir=None):
    """
    Plot PCA for features at different depths.
    features_dict: {depth_label: features_array}
    """
    n_layers = len(features_dict)
    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 5))
    if n_layers == 1:
        axes = [axes]

    if isinstance(labels, np.ndarray):
        labs = labels
    else:
        labs = labels.numpy()

    n_classes = len(class_names)
    cmap = plt.cm.get_cmap("tab20", n_classes)

    for ax, (depth_label, feats) in zip(axes, features_dict.items()):
        if not isinstance(feats, np.ndarray):
            feats = feats.numpy()
        pca = PCA(n_components=2, random_state=config.SEED)
        embedded = pca.fit_transform(feats)
        var = pca.explained_variance_ratio_

        for i in range(n_classes):
            mask = labs == i
            if mask.sum() > 0:
                ax.scatter(embedded[mask, 0], embedded[mask, 1],
                           c=[cmap(i)], s=8, alpha=0.5)
        ax.set_title(f"{depth_label}\n(Var: {sum(var):.1%})", fontsize=10)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    plt.suptitle(title_prefix, fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path_dir:
        os.makedirs(save_path_dir, exist_ok=True)
        path = os.path.join(save_path_dir, "pca_across_layers.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved PCA across layers → {path}")
    plt.close()
