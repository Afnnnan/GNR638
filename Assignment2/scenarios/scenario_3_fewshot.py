"""
Scenario 4.3 — Few-Shot Learning Analysis
Evaluate data efficiency under 100%, 20%, 5% training data.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import dataset
import models
import train as trainer
import metrics


def run(model_names=None, smoke_test=False):
    """Run Scenario 3: Few-Shot Learning for all models."""
    config.set_seed()
    model_names = model_names or config.MODEL_NAMES
    scenario_dir = os.path.join(config.RESULTS_DIR, "scenario_3_fewshot")
    os.makedirs(scenario_dir, exist_ok=True)

    all_results = []
    fractions = config.FEWSHOT_FRACTIONS  # [1.0, 0.2, 0.05]

    for mname in model_names:
        print(f"\n{'='*70}")
        print(f"  Scenario 3 — Few-Shot Learning | Model: {mname}")
        print(f"{'='*70}")

        model_dir = os.path.join(scenario_dir, mname)
        os.makedirs(model_dir, exist_ok=True)

        model_accs = {}
        model_train_val_gaps = {}

        for frac in fractions:
            pct_label = f"{frac*100:.0f}%"
            print(f"\n  --- Data fraction: {pct_label} ---")
            config.set_seed()

            epochs = (2 if smoke_test else
                      (config.EPOCHS_FULL if frac == 1.0 else config.EPOCHS_FEWSHOT))

            # Build data loaders with few-shot fraction
            train_loader, val_loader, class_names = dataset.build_dataloaders(
                fewshot_fraction=frac
            )

            # Build model — full fine-tuning for few-shot experiments
            model, model_info = models.get_model(mname, strategy="full")

            # FLOPs
            flops_info = trainer.compute_flops_params(model)

            # Checkpoint
            ckpt_path = os.path.join(config.CHECKPOINT_DIR,
                                      f"s3_{mname}_fewshot_{pct_label}.pth")

            # Train
            history = trainer.train_model(
                model, train_loader, val_loader,
                epochs=epochs,
                save_path=ckpt_path,
            )

            model_accs[pct_label] = history["best_val_acc"]
            # Train-val gap at best epoch
            best_epoch_idx = history["val_acc"].index(max(history["val_acc"]))
            train_val_gap = history["train_acc"][best_epoch_idx] - history["val_acc"][best_epoch_idx]
            model_train_val_gaps[pct_label] = train_val_gap

            # Accuracy curves
            metrics.plot_accuracy_curves(
                history, title=f"{mname} — {pct_label} data",
                save_path=os.path.join(model_dir, f"accuracy_curves_{pct_label}.png")
            )

            all_results.append([
                mname, pct_label,
                f"{model_info['trainable_params']:,}",
                flops_info["MACs_str"],
                f"{history['best_val_acc']:.2f}%",
                f"{train_val_gap:.2f}%",
            ])

        # Compute relative performance drop
        if "100%" in model_accs and "5%" in model_accs:
            delta = (model_accs["100%"] - model_accs["5%"]) / model_accs["100%"]
            print(f"  Δ (relative drop) for {mname}: {delta:.4f} ({delta*100:.2f}%)")
            all_results.append([mname, "Δ (drop)", "—", "—",
                                f"{delta*100:.2f}%", "—"])

    # Summary table
    metrics.create_results_table(
        all_results,
        columns=["Model", "Data %", "Trainable Params", "MACs",
                 "Best Val Acc", "Train-Val Gap"],
        title="Scenario 3 — Few-Shot Learning Results",
        save_path=os.path.join(scenario_dir, "summary_table.png")
    )

    print("\n✓ Scenario 3 complete!")
    return all_results


if __name__ == "__main__":
    smoke = "--smoke-test" in sys.argv
    run(smoke_test=smoke)
