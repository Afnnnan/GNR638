"""
Scenario 4.5 — Layer-Wise Feature Probing
Extract intermediate representations and train linear classifiers per layer.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import config
import dataset
import models
import metrics
import visualize


def extract_features_at_layers(model, model_name, loader, device=None):
    """
    Extract features at early/middle/mid_deep/final layers using hooks.
    Returns: {depth_label: (features_array, labels_array)}
    """
    device = device or config.DEVICE
    model = model.to(device)
    model.eval()

    extractor = models.FeatureExtractor(model, model_name)

    all_features = {k: [] for k in models.get_layer_names_for_probing(model_name).keys()}
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            extractor.clear_features()
            _ = model(images)
            feats = extractor.get_features()
            for depth_label, feat_tensor in feats.items():
                all_features[depth_label].append(feat_tensor.cpu().numpy())
            all_labels.append(labels.numpy())

    extractor.remove_hooks()

    result = {}
    labels_all = np.concatenate(all_labels)
    for depth_label, feat_list in all_features.items():
        if feat_list:
            result[depth_label] = (np.concatenate(feat_list), labels_all)
    return result


def train_linear_probe_sklearn(X_train, y_train, X_val, y_val):
    """Train a logistic regression classifier and return val accuracy."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    clf = LogisticRegression(max_iter=1000, random_state=config.SEED, n_jobs=-1)
    clf.fit(X_train_s, y_train)
    val_acc = 100.0 * clf.score(X_val_s, y_val)
    return val_acc


def get_fixed_subset_loader(n_per_class=30, seed=config.SEED):
    """
    Create a fixed subset: 30 samples per class for PCA visualization.
    Returns loader, subset_indices, class_names.
    """
    full_dataset = dataset.get_dataset(transform=dataset.get_val_transform())
    targets = np.array([s[1] for s in full_dataset.samples])
    rng = np.random.RandomState(seed)

    selected_indices = []
    for cls_idx in range(config.NUM_CLASSES):
        cls_indices = np.where(targets == cls_idx)[0]
        if len(cls_indices) >= n_per_class:
            chosen = rng.choice(cls_indices, size=n_per_class, replace=False)
        else:
            chosen = cls_indices
        selected_indices.extend(chosen)

    subset = Subset(full_dataset, selected_indices)
    loader = DataLoader(subset, batch_size=config.BATCH_SIZE, shuffle=False,
                        num_workers=config.NUM_WORKERS, pin_memory=True)
    return loader, full_dataset.classes


def run(model_names=None, smoke_test=False):
    """Run Scenario 5: Layer-Wise Feature Probing."""
    config.set_seed()
    model_names = model_names or config.MODEL_NAMES
    scenario_dir = os.path.join(config.RESULTS_DIR, "scenario_5_layerwise")
    os.makedirs(scenario_dir, exist_ok=True)

    # Build data loaders
    train_loader, val_loader, class_names = dataset.build_dataloaders()

    # Fixed subset for PCA visualization
    fixed_loader, _ = get_fixed_subset_loader()

    layer_accuracy_results = {}   # {model_name: {depth: acc}}
    layer_norm_results = {}       # {model_name: {depth: (mean, std)}}

    all_table_rows = []

    for mname in model_names:
        print(f"\n{'='*70}")
        print(f"  Scenario 5 — Layer-Wise Probing | Model: {mname}")
        print(f"{'='*70}")

        model_dir = os.path.join(scenario_dir, mname)
        os.makedirs(model_dir, exist_ok=True)

        # Load pretrained model (frozen, for feature extraction)
        model, model_info = models.get_model(mname, strategy="linear_probe")

        # Try loading fine-tuned checkpoint if available
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"s2_{mname}_full.pth")
        if os.path.exists(ckpt_path):
            # We use the pretrained model for probing (not fine-tuned),
            # but could also compare. Using pretrained for layer analysis.
            pass

        # Extract features from train and val sets
        print("  Extracting train features...")
        train_feats = extract_features_at_layers(model, mname, train_loader)
        print("  Extracting val features...")
        val_feats = extract_features_at_layers(model, mname, val_loader)

        depth_accs = {}
        depth_norms = {}

        for depth_label in train_feats.keys():
            X_train, y_train = train_feats[depth_label]
            X_val, y_val = val_feats[depth_label]

            # Feature norm statistics
            norms = np.linalg.norm(X_val, axis=1)
            mean_norm, std_norm = norms.mean(), norms.std()
            depth_norms[depth_label] = (mean_norm, std_norm)

            # Train linear probe
            print(f"    Layer '{depth_label}': dim={X_train.shape[1]}, "
                  f"norm={mean_norm:.2f}±{std_norm:.2f}")
            acc = train_linear_probe_sklearn(X_train, y_train, X_val, y_val)
            depth_accs[depth_label] = acc
            print(f"    → Val Acc: {acc:.2f}%")

            all_table_rows.append([mname, depth_label, X_train.shape[1],
                                   f"{mean_norm:.2f}±{std_norm:.2f}",
                                   f"{acc:.2f}%"])

        layer_accuracy_results[mname] = depth_accs
        layer_norm_results[mname] = depth_norms

        # PCA across layers for this model (on fixed subset)
        print("  Extracting fixed subset features for PCA...")
        fixed_feats = extract_features_at_layers(model, mname, fixed_loader)
        pca_feats = {k: v[0] for k, v in fixed_feats.items()}
        pca_labels = list(fixed_feats.values())[0][1]

        visualize.plot_pca_across_layers(
            pca_feats, pca_labels, class_names,
            title_prefix=f"PCA Across Layers — {mname}",
            save_path_dir=model_dir
        )

    # ── Cross-model plots ──
    # Accuracy vs depth
    visualize.plot_layerwise_accuracy(
        layer_accuracy_results,
        title="Layer-Wise Probing Accuracy",
        save_path=os.path.join(scenario_dir, "layerwise_accuracy.png")
    )

    # Feature norm comparison
    visualize.plot_feature_norms(
        layer_norm_results,
        title="Feature Norm Statistics Across Layers",
        save_path=os.path.join(scenario_dir, "feature_norms.png")
    )

    # Summary table
    metrics.create_results_table(
        all_table_rows,
        columns=["Model", "Layer", "Feature Dim", "Norm (μ±σ)", "Val Acc"],
        title="Scenario 5 — Layer-Wise Feature Probing Results",
        save_path=os.path.join(scenario_dir, "summary_table.png")
    )

    print("\n✓ Scenario 5 complete!")
    return all_table_rows


if __name__ == "__main__":
    smoke = "--smoke-test" in sys.argv
    run(smoke_test=smoke)
