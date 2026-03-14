"""
Scenario 4.1 — Linear Probe Transfer
Freeze all backbone parameters, train only the linear classifier.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import dataset
import models
import train as trainer
import metrics
import visualize


def run(model_names=None, epochs=None, smoke_test=False):
    """Run Scenario 1: Linear Probe Transfer for all models."""
    config.set_seed()
    model_names = model_names or config.MODEL_NAMES
    epochs = epochs or (2 if smoke_test else config.EPOCHS_FULL)
    scenario_dir = os.path.join(config.RESULTS_DIR, "scenario_1_linear_probe")
    os.makedirs(scenario_dir, exist_ok=True)

    results_summary = []

    for mname in model_names:
        print(f"\n{'='*70}")
        print(f"  Scenario 1 — Linear Probe | Model: {mname}")
        print(f"{'='*70}")

        model_dir = os.path.join(scenario_dir, mname)
        os.makedirs(model_dir, exist_ok=True)

        # Build data loaders
        train_loader, val_loader, class_names = dataset.build_dataloaders()

        # Build model with linear probe strategy
        model, model_info = models.get_model(mname, strategy="linear_probe")

        # Compute FLOPs/MACs
        flops_info = trainer.compute_flops_params(model)

        # Save checkpoint path
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"s1_{mname}_linear_probe.pth")

        # Train
        history = trainer.train_model(
            model, train_loader, val_loader,
            epochs=epochs,
            save_path=ckpt_path,
        )

        # ── Plots ──
        # 1. Accuracy curves
        metrics.plot_accuracy_curves(
            history, title=f"Linear Probe — {mname}",
            save_path=os.path.join(model_dir, "accuracy_curves.png")
        )

        # 2. Confusion matrix
        # Load best model
        model.load_state_dict(
            __import__("torch").load(ckpt_path, map_location=config.DEVICE, weights_only=True)
        )
        y_pred, y_true = trainer.get_predictions(model, val_loader)
        metrics.plot_confusion_matrix(
            y_true, y_pred, class_names,
            title=f"Confusion Matrix — Linear Probe — {mname}",
            save_path=os.path.join(model_dir, "confusion_matrix.png")
        )

        # 3. t-SNE / PCA of feature embeddings
        feats, labels = trainer.extract_features(model, val_loader)
        visualize.plot_tsne(
            feats, labels, class_names,
            title=f"t-SNE — Linear Probe — {mname}",
            save_path=os.path.join(model_dir, "tsne.png")
        )
        visualize.plot_pca(
            feats, labels, class_names,
            title=f"PCA — Linear Probe — {mname}",
            save_path=os.path.join(model_dir, "pca.png")
        )

        results_summary.append([
            mname, f"{model_info['total_params']:,}",
            f"{model_info['trainable_params']:,}",
            f"{model_info['pct_trainable']:.1f}%",
            flops_info["MACs_str"], flops_info["FLOPs_str"],
            f"{history['best_val_acc']:.2f}%",
        ])

    # Summary table
    metrics.create_results_table(
        results_summary,
        columns=["Model", "Total Params", "Trainable Params", "% Trainable",
                 "MACs", "FLOPs", "Best Val Acc"],
        title="Scenario 1 — Linear Probe Transfer Results",
        save_path=os.path.join(scenario_dir, "summary_table.png")
    )

    print("\n✓ Scenario 1 complete!")
    return results_summary


if __name__ == "__main__":
    smoke = "--smoke-test" in sys.argv
    run(smoke_test=smoke)
