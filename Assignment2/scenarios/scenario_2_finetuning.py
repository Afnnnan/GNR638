"""
Scenario 4.2 — Fine-Tuning Strategies
Compare: linear probe, last block, full, selective (20%) fine-tuning.
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


STRATEGIES = ["linear_probe", "last_block", "full", "selective"]


def run(model_names=None, epochs=None, smoke_test=False):
    """Run Scenario 2: Fine-Tuning Strategies for all models."""
    config.set_seed()
    model_names = model_names or config.MODEL_NAMES
    epochs = epochs or (2 if smoke_test else config.EPOCHS_FULL)
    scenario_dir = os.path.join(config.RESULTS_DIR, "scenario_2_finetuning")
    os.makedirs(scenario_dir, exist_ok=True)

    all_results = []

    for mname in model_names:
        print(f"\n{'='*70}")
        print(f"  Scenario 2 — Fine-Tuning Strategies | Model: {mname}")
        print(f"{'='*70}")

        model_dir = os.path.join(scenario_dir, mname)
        os.makedirs(model_dir, exist_ok=True)

        histories = {}
        grad_norms_all = {}
        acc_vs_unfrozen = []

        for strategy in STRATEGIES:
            print(f"\n  --- Strategy: {strategy} ---")
            config.set_seed()

            # Build data loaders (fresh each time for reproducibility)
            train_loader, val_loader, class_names = dataset.build_dataloaders()

            # Build model
            model, model_info = models.get_model(mname, strategy=strategy)

            # Compute FLOPs
            flops_info = trainer.compute_flops_params(model)

            # Checkpoint
            ckpt_path = os.path.join(config.CHECKPOINT_DIR,
                                      f"s2_{mname}_{strategy}.pth")

            # Train with gradient norm tracking
            history = trainer.train_model(
                model, train_loader, val_loader,
                epochs=epochs,
                save_path=ckpt_path,
                track_grad_norms=True,
            )

            histories[strategy] = history

            # Track gradient norms (last epoch)
            if history["grad_norms"]:
                grad_norms_all[strategy] = history["grad_norms"][-1]

            # For accuracy vs % unfrozen plot
            acc_vs_unfrozen.append((
                model_info["pct_trainable"],
                history["best_val_acc"],
                strategy
            ))

            # Accuracy curves per strategy
            metrics.plot_accuracy_curves(
                history, title=f"{mname} — {strategy}",
                save_path=os.path.join(model_dir, f"accuracy_curves_{strategy}.png")
            )

            all_results.append([
                mname, strategy,
                f"{model_info['total_params']:,}",
                f"{model_info['trainable_params']:,}",
                f"{model_info['pct_trainable']:.1f}%",
                flops_info["MACs_str"], flops_info["FLOPs_str"],
                f"{history['best_val_acc']:.2f}%",
            ])

        # ── Cross-strategy plots for this model ──
        # 1. Convergence comparison
        metrics.plot_convergence_comparison(
            histories, title=f"Convergence — {mname}",
            save_path=os.path.join(model_dir, "convergence_comparison.png")
        )

        # 2. Gradient norm comparison
        if grad_norms_all:
            visualize.plot_gradient_norms_comparison(
                grad_norms_all, title=f"Gradient Norms — {mname}",
                save_path=os.path.join(model_dir, "gradient_norms_comparison.png")
            )

        # 3. Accuracy vs % unfrozen parameters
        visualize.plot_accuracy_vs_unfrozen(
            acc_vs_unfrozen, title=f"Accuracy vs Unfrozen Params — {mname}",
            save_path=os.path.join(model_dir, "accuracy_vs_unfrozen.png")
        )

    # Summary table
    metrics.create_results_table(
        all_results,
        columns=["Model", "Strategy", "Total Params", "Trainable Params",
                 "% Trainable", "MACs", "FLOPs", "Best Val Acc"],
        title="Scenario 2 — Fine-Tuning Strategies Results",
        save_path=os.path.join(scenario_dir, "summary_table.png")
    )

    print("\n✓ Scenario 2 complete!")
    return all_results


if __name__ == "__main__":
    smoke = "--smoke-test" in sys.argv
    run(smoke_test=smoke)
