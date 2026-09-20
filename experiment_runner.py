"""
Automated experiment runner for CamylaNet architecture comparison.

Workflow:
  1. Smoke test (1 epoch) for each architecture -> verify pipeline
  2. Full training (N epochs) for selected architectures
  3. Evaluation + results summary

Usage:
  # Smoke test all 5 architectures
  python experiment_runner.py --dataset_id 101 --mode smoke_test

  # Full training for top 3 (specify after smoke test)
  python experiment_runner.py --dataset_id 101 --mode full \
      --architectures nnUNet SwinUNETR SegResNet --epochs 100

  # Include foundation model
  python experiment_runner.py --dataset_id 101 --mode smoke_test \
      --include_foundation
"""

import os
import sys
import json
import time
import argparse
import traceback
from datetime import datetime

RESULTS_FILE = "experiment_results.json"

ARCHITECTURE_MAP = {
    "nnUNet": "nnUNetTrainer",
    "SwinUNETR": "SwinUNETRTrainer",
    "SegResNet": "SegResNetTrainer",
    "UNetPlusPlus": "UNetPlusPlusTrainer",
    "UMamba": "UMambaTrainer",
}

DEFAULT_ARCHITECTURES = ["nnUNet", "SwinUNETR", "SegResNet", "UNetPlusPlus", "UMamba"]


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {"experiments": [], "last_updated": None}


def save_results(results):
    results["last_updated"] = datetime.now().isoformat()
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def run_smoke_test(dataset_id, architecture_name, trainer_class_name,
                   plans_identifier, fold=0):
    """Run 1-epoch smoke test for a single architecture."""
    import camylanet

    exp_name = f"smoke_{architecture_name}_fold{fold}"
    print(f"\n{'='*60}")
    print(f"SMOKE TEST: {architecture_name} ({trainer_class_name})")
    print(f"  Dataset: {dataset_id}, Fold: {fold}, Exp: {exp_name}")
    print(f"{'='*60}")

    start_time = time.time()
    try:
        result_folder, training_log = camylanet.training_network_1epoch(
            dataset_id=dataset_id,
            configuration='2d',
            trainer_class=trainer_class_name,
            plans_identifier=plans_identifier,
            exp_name=exp_name,
        )
        elapsed = time.time() - start_time

        train_loss = training_log.get('train_losses', [None])
        val_loss = training_log.get('val_losses', [None])
        train_loss_val = train_loss[0] if train_loss else None
        val_loss_val = val_loss[0] if val_loss else None

        print(f"  Train loss: {train_loss_val}")
        print(f"  Val loss: {val_loss_val}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Result folder: {result_folder}")

        return {
            "architecture": architecture_name,
            "trainer_class": trainer_class_name,
            "mode": "smoke_test",
            "fold": fold,
            "status": "pass",
            "train_loss": train_loss_val,
            "val_loss": val_loss_val,
            "elapsed_seconds": elapsed,
            "result_folder": result_folder,
            "exp_name": exp_name,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = str(e)
        tb = traceback.format_exc()
        print(f"  FAILED: {error_msg}")
        print(f"  Time: {elapsed:.1f}s")
        return {
            "architecture": architecture_name,
            "trainer_class": trainer_class_name,
            "mode": "smoke_test",
            "fold": fold,
            "status": "fail",
            "error": error_msg,
            "traceback": tb[-500:],
            "elapsed_seconds": elapsed,
            "exp_name": exp_name,
            "timestamp": datetime.now().isoformat(),
        }


def run_full_training(dataset_id, architecture_name, trainer_class_name,
                      plans_identifier, fold=0, num_epochs=100):
    """Run full training for a single architecture."""
    import camylanet

    exp_name = f"full_{architecture_name}_fold{fold}"
    print(f"\n{'='*60}")
    print(f"FULL TRAINING: {architecture_name} ({trainer_class_name})")
    print(f"  Dataset: {dataset_id}, Fold: {fold}, Epochs: {num_epochs}")
    print(f"  Exp: {exp_name}")
    print(f"{'='*60}")

    start_time = time.time()
    try:
        result_folder, training_log = camylanet.training_network(
            dataset_id=dataset_id,
            configuration='2d',
            trainer_class=trainer_class_name,
            plans_identifier=plans_identifier,
            num_epochs=num_epochs,
            exp_name=exp_name,
        )
        elapsed = time.time() - start_time

        train_losses = training_log.get('train_losses', [])
        val_losses = training_log.get('val_losses', [])

        print(f"  Epochs trained: {len(train_losses)}")
        print(f"  Final train loss: {train_losses[-1] if train_losses else 'N/A'}")
        print(f"  Final val loss: {val_losses[-1] if val_losses else 'N/A'}")
        print(f"  Total time: {elapsed:.1f}s ({elapsed/3600:.2f}h)")

        # Evaluate
        print(f"  Evaluating...")
        eval_results = camylanet.evaluate(
            dataset_id=dataset_id,
            result_folder=result_folder,
            exp_name=exp_name,
        )

        dice = eval_results.get('foreground_mean', {}).get('Dice', None)
        iou = eval_results.get('foreground_mean', {}).get('IoU', None)
        print(f"  Mean Dice: {dice}")
        print(f"  Mean IoU: {iou}")

        # Per-class Dice
        class_dice = {}
        for label_name, metrics in eval_results.items():
            if isinstance(metrics, dict) and 'Dice' in metrics:
                class_dice[label_name] = metrics['Dice']

        return {
            "architecture": architecture_name,
            "trainer_class": trainer_class_name,
            "mode": "full",
            "fold": fold,
            "status": "pass",
            "num_epochs": len(train_losses),
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_val_loss": val_losses[-1] if val_losses else None,
            "mean_dice": dice,
            "mean_iou": iou,
            "per_class_dice": class_dice,
            "elapsed_seconds": elapsed,
            "elapsed_hours": elapsed / 3600,
            "result_folder": result_folder,
            "exp_name": exp_name,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = str(e)
        tb = traceback.format_exc()
        print(f"  FAILED: {error_msg}")
        print(f"  Time: {elapsed:.1f}s")
        return {
            "architecture": architecture_name,
            "trainer_class": trainer_class_name,
            "mode": "full",
            "fold": fold,
            "status": "fail",
            "error": error_msg,
            "traceback": tb[-500:],
            "elapsed_seconds": elapsed,
            "exp_name": exp_name,
            "timestamp": datetime.now().isoformat(),
        }


def run_foundation_smoke_test(dataset_id, plans_identifier, fold=0,
                              model_name="UNI"):
    """Run smoke test for foundation model (UNI/CONCH)."""
    from foundation_model_trainer import FoundationModelTrainer

    arch_name = f"FoundationModel_{model_name}"
    exp_name = f"smoke_{arch_name}_fold{fold}"
    print(f"\n{'='*60}")
    print(f"SMOKE TEST (Foundation): {arch_name}")
    print(f"  Dataset: {dataset_id}, Fold: {fold}")
    print(f"{'='*60}")

    import camylanet

    start_time = time.time()
    try:
        result_folder, training_log = camylanet.training_network_1epoch(
            dataset_id=dataset_id,
            configuration='2d',
            trainer_class=FoundationModelTrainer,
            plans_identifier=plans_identifier,
            exp_name=exp_name,
        )
        elapsed = time.time() - start_time

        train_loss = training_log.get('train_losses', [None])
        val_loss = training_log.get('val_losses', [None])

        print(f"  Train loss: {train_loss[0] if train_loss else 'N/A'}")
        print(f"  Val loss: {val_loss[0] if val_loss else 'N/A'}")
        print(f"  Time: {elapsed:.1f}s")

        return {
            "architecture": arch_name,
            "trainer_class": "FoundationModelTrainer",
            "mode": "smoke_test",
            "fold": fold,
            "status": "pass",
            "train_loss": train_loss[0] if train_loss else None,
            "val_loss": val_loss[0] if val_loss else None,
            "elapsed_seconds": elapsed,
            "exp_name": exp_name,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  FAILED: {e}")
        return {
            "architecture": arch_name,
            "trainer_class": "FoundationModelTrainer",
            "mode": "smoke_test",
            "fold": fold,
            "status": "fail",
            "error": str(e),
            "elapsed_seconds": elapsed,
            "exp_name": exp_name,
            "timestamp": datetime.now().isoformat(),
        }


def print_summary(results):
    """Print summary table of all experiments."""
    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"{'Architecture':<25} {'Mode':<12} {'Status':<8} {'Dice':<10} {'Time':<10}")
    print(f"{'-'*70}")

    for exp in results.get("experiments", []):
        arch = exp.get("architecture", "?")
        mode = exp.get("mode", "?")
        status = exp.get("status", "?")
        dice = exp.get("mean_dice") or exp.get("val_loss", "N/A")
        if isinstance(dice, float):
            dice_str = f"{dice:.4f}"
        else:
            dice_str = str(dice)
        elapsed = exp.get("elapsed_seconds", 0)
        time_str = f"{elapsed:.0f}s" if elapsed < 3600 else f"{elapsed/3600:.2f}h"
        print(f"{arch:<25} {mode:<12} {status:<8} {dice_str:<10} {time_str:<10}")

    print(f"{'='*70}")
    print(f"Last updated: {results.get('last_updated', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(
        description='Automated CamylaNet architecture comparison runner')
    parser.add_argument('--dataset_id', type=int, default=101,
                        help='nnU-Net dataset ID')
    parser.add_argument('--mode', type=str, default='smoke_test',
                        choices=['smoke_test', 'full'],
                        help='smoke_test (1 epoch) or full training')
    parser.add_argument('--architectures', nargs='+',
                        default=DEFAULT_ARCHITECTURES,
                        help='Architectures to test')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs for full training')
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold number (0-4)')
    parser.add_argument('--include_foundation', action='store_true',
                        help='Include UNI foundation model in comparison')
    parser.add_argument('--plans_identifier', type=str, default='nnUNetPlans',
                        help='Plans identifier from plan_and_preprocess')
    args = parser.parse_args()

    import camylanet

    # Ensure preprocessing is done
    print(f"Ensuring preprocessing for dataset {args.dataset_id}...")
    plans_identifier = args.plans_identifier
    try:
        plans_identifier = camylanet.plan_and_preprocess(
            dataset_id=args.dataset_id,
            configurations=['2d']
        )
        print(f"Plans: {plans_identifier}")
    except Exception as e:
        print(f"Preprocessing note: {e}")
        print(f"Using plans_identifier={plans_identifier}")

    results = load_results()

    # Run experiments
    for arch_name in args.architectures:
        trainer_name = ARCHITECTURE_MAP.get(arch_name, arch_name)

        if args.mode == 'smoke_test':
            result = run_smoke_test(
                args.dataset_id, arch_name, trainer_name,
                plans_identifier, args.fold
            )
        else:
            result = run_full_training(
                args.dataset_id, arch_name, trainer_name,
                plans_identifier, args.fold, args.epochs
            )

        results["experiments"].append(result)
        save_results(results)

    # Foundation model track
    if args.include_foundation:
        result = run_foundation_smoke_test(
            args.dataset_id, plans_identifier, args.fold
        )
        results["experiments"].append(result)
        save_results(results)

    print_summary(results)
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == '__main__':
    main()
