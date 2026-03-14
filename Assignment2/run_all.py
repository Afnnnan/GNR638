"""
Master runner script for GNR 638 Assignment 2.
Runs all experimental scenarios sequentially.

Usage:
    python run_all.py                     # Run all scenarios (full training)
    python run_all.py --smoke-test        # Quick test (2 epochs, validates pipeline)
    python run_all.py --scenario 1        # Run only scenario 1
    python run_all.py --scenario 1 2 5    # Run scenarios 1, 2, and 5
"""
import os
import sys
import argparse
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config


def main():
    parser = argparse.ArgumentParser(description="GNR 638 Assignment 2 — Run Experiments")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick pipeline test (2 epochs)")
    parser.add_argument("--scenario", nargs="+", type=int, default=None,
                        help="Run specific scenarios (e.g., --scenario 1 2 5)")
    args = parser.parse_args()

    config.set_seed()

    scenarios_to_run = args.scenario or [1, 2, 3, 4, 5]
    smoke = args.smoke_test

    print(f"\n{'#'*70}")
    print(f"  GNR 638 Assignment 2 — Pre-trained CNN Backbone Analysis")
    print(f"  Device: {config.DEVICE}")
    print(f"  Seed: {config.SEED}")
    print(f"  Models: {config.MODEL_NAMES}")
    print(f"  Scenarios: {scenarios_to_run}")
    print(f"  Smoke test: {smoke}")
    print(f"{'#'*70}\n")

    total_start = time.time()

    if 1 in scenarios_to_run:
        print("\n" + "="*70)
        print("  SCENARIO 1: Linear Probe Transfer")
        print("="*70)
        from scenarios import scenario_1_linear_probe
        scenario_1_linear_probe.run(smoke_test=smoke)

    if 2 in scenarios_to_run:
        print("\n" + "="*70)
        print("  SCENARIO 2: Fine-Tuning Strategies")
        print("="*70)
        from scenarios import scenario_2_finetuning
        scenario_2_finetuning.run(smoke_test=smoke)

    if 3 in scenarios_to_run:
        print("\n" + "="*70)
        print("  SCENARIO 3: Few-Shot Learning Analysis")
        print("="*70)
        from scenarios import scenario_3_fewshot
        scenario_3_fewshot.run(smoke_test=smoke)

    if 4 in scenarios_to_run:
        print("\n" + "="*70)
        print("  SCENARIO 4: Corruption Robustness Evaluation")
        print("="*70)
        from scenarios import scenario_4_corruption
        scenario_4_corruption.run(smoke_test=smoke)

    if 5 in scenarios_to_run:
        print("\n" + "="*70)
        print("  SCENARIO 5: Layer-Wise Feature Probing")
        print("="*70)
        from scenarios import scenario_5_layerwise_probe
        scenario_5_layerwise_probe.run(smoke_test=smoke)

    elapsed = time.time() - total_start
    hours, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    print(f"\n{'#'*70}")
    print(f"  All requested scenarios complete!")
    print(f"  Total time: {int(hours)}h {int(mins)}m {secs:.1f}s")
    print(f"  Results saved to: {config.RESULTS_DIR}")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
