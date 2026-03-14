"""
Scenario 4.4 — Corruption Robustness Evaluation
Evaluate under Gaussian noise, motion blur, and brightness shift.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import config
import dataset
import models
import train as trainer
import metrics


def run(model_names=None, smoke_test=False):
    """Run Scenario 4: Corruption Robustness for all models."""
    config.set_seed()
    model_names = model_names or config.MODEL_NAMES
    scenario_dir = os.path.join(config.RESULTS_DIR, "scenario_4_corruption")
    os.makedirs(scenario_dir, exist_ok=True)

    # Define corruptions
    corruptions = []
    for sigma in config.GAUSSIAN_NOISE_SIGMAS:
        corruptions.append(("gaussian_noise", f"Gaussian σ={sigma}", {"sigma": sigma}))
    corruptions.append(("motion_blur", "Motion Blur (k=15)", {"kernel_size": 15}))
    corruptions.append(("brightness_shift", "Brightness ×1.5", {"factor": 1.5}))

    all_results = []

    for mname in model_names:
        print(f"\n{'='*70}")
        print(f"  Scenario 4 — Corruption Robustness | Model: {mname}")
        print(f"{'='*70}")

        model_dir = os.path.join(scenario_dir, mname)
        os.makedirs(model_dir, exist_ok=True)

        # Load best fine-tuned model from Scenario 2 (full fine-tuning)
        model, model_info = models.get_model(mname, strategy="full", pretrained=True)
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"s2_{mname}_full.pth")

        if os.path.exists(ckpt_path):
            model.load_state_dict(
                torch.load(ckpt_path, map_location=config.DEVICE, weights_only=True)
            )
            print(f"  Loaded checkpoint: {ckpt_path}")
        else:
            print(f"  ⚠ No checkpoint found at {ckpt_path}. Using pretrained weights.")
            print(f"    Run Scenario 2 first, or training on the fly...")
            # Quick train if no checkpoint exists
            train_loader, val_loader, class_names = dataset.build_dataloaders()
            epochs = 2 if smoke_test else config.EPOCHS_FULL
            history = trainer.train_model(
                model, train_loader, val_loader,
                epochs=epochs, save_path=ckpt_path,
            )
            model.load_state_dict(
                torch.load(ckpt_path, map_location=config.DEVICE, weights_only=True)
            )

        # Evaluate on clean validation set first
        _, clean_val_loader, class_names = dataset.build_dataloaders()
        _, clean_acc = trainer.evaluate(model, clean_val_loader)
        print(f"  Clean accuracy: {clean_acc:.2f}%")

        all_results.append([mname, "Clean", f"{clean_acc:.2f}%", "—", "—"])

        # Evaluate under each corruption
        for corr_type, corr_label, corr_kwargs in corruptions:
            print(f"\n  --- Corruption: {corr_label} ---")
            corr_loader = dataset.build_corruption_loader(
                corruption_type=corr_type, **corr_kwargs
            )
            _, corr_acc = trainer.evaluate(model, corr_loader)

            corruption_error = 1.0 - (corr_acc / 100.0)
            relative_robustness = corr_acc / clean_acc if clean_acc > 0 else 0

            print(f"    Corrupted Acc: {corr_acc:.2f}% | "
                  f"Corruption Error: {corruption_error:.4f} | "
                  f"Relative Robustness: {relative_robustness:.4f}")

            all_results.append([
                mname, corr_label,
                f"{corr_acc:.2f}%",
                f"{corruption_error:.4f}",
                f"{relative_robustness:.4f}",
            ])

    # Summary table
    metrics.create_results_table(
        all_results,
        columns=["Model", "Corruption", "Accuracy", "Corruption Error",
                 "Relative Robustness"],
        title="Scenario 4 — Corruption Robustness Results",
        save_path=os.path.join(scenario_dir, "summary_table.png")
    )

    print("\n✓ Scenario 4 complete!")
    return all_results


if __name__ == "__main__":
    smoke = "--smoke-test" in sys.argv
    run(smoke_test=smoke)
