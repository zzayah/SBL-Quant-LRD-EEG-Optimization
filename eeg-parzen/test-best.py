#!/usr/bin/env python3
"""Retrain the three NAS winners across all three EEG datasets."""

import argparse
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import optuna
import torch
from torch import nn


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cnn
from benchmarking.evaluation import evaluate
from data.eeg_dataset import (
    SUPPORTED_DATASETS,
    make_data_loaders,
    make_final_data_loaders,
)


SOURCE_DATASETS = tuple(SUPPORTED_DATASETS)
CALIBRATION_MAX_EPOCHS = 100
CALIBRATION_PATIENCE = 15


def load_best(path: str | Path, expected_dataset: str) -> dict:
    """Load the completed trial with the highest validation objective."""
    with Path(path).open(encoding="utf-8") as file:
        entries = [json.loads(line) for line in file if line.strip()]
    entries = [entry for entry in entries if entry.get("dataset") == expected_dataset]
    if not entries:
        raise RuntimeError(f"No completed {expected_dataset} trials found in {path}")
    return max(entries, key=lambda entry: float(entry["objective"]["value"]))


def train_final(
    model: nn.Module,
    train_loader,
    device: torch.device,
    epochs: int,
) -> float:
    """Train on all development subjects for the calibrated epoch budget."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cnn.TRAINING["learning_rate"],
        weight_decay=cnn.TRAINING["weight_decay"],
    )
    started = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for signals, targets in train_loader:
            signals = signals.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(signals), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Final training loss became non-finite")
            loss.backward()
            optimizer.step()
    return time.perf_counter() - started


def calibrate_epochs(
    architecture: dict,
    dataset: str,
    seed: int,
    device: torch.device,
) -> tuple[int, list[dict]]:
    """Choose a final epoch count using only training and validation subjects."""
    folds = range(3) if dataset == "bnci2014_001" else [None]
    fold_results = []
    for fold in folds:
        cnn.seed_everything(seed)
        loaders, info = make_data_loaders(
            dataset,
            batch_size=cnn.TRAINING["batch_size"],
            seed=seed,
            fold=fold,
        )
        model = cnn.EEGCNN(info["channels"], architecture).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cnn.TRAINING["learning_rate"],
            weight_decay=cnn.TRAINING["weight_decay"],
        )
        best_epoch = 0
        best_score = -1.0
        stale_epochs = 0
        for epoch in range(1, CALIBRATION_MAX_EPOCHS + 1):
            model.train()
            for signals, targets in loaders["train"]:
                signals = signals.to(device)
                targets = targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(signals), targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Calibration loss became non-finite")
                loss.backward()
                optimizer.step()
            metrics = cnn.evaluate_model(
                model, loaders["validation"], device, criterion
            )
            if metrics["balanced_accuracy"] > best_score:
                best_score = float(metrics["balanced_accuracy"])
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= CALIBRATION_PATIENCE:
                break
        fold_results.append(
            {
                "fold": fold,
                "best_epoch": best_epoch,
                "best_validation_balanced_accuracy": best_score,
                "epochs_run": epoch,
            }
        )
    selected_epochs = int(np.median([result["best_epoch"] for result in fold_results]))
    return max(selected_epochs, 1), fold_results


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}


def main(
    trial_files: dict[str, str],
    output_dir: str,
    run_id: str,
    seed: int,
) -> None:
    """Run the predetermined three-winner by three-dataset test matrix."""
    winners = {
        dataset: load_best(trial_files[dataset], dataset) for dataset in SOURCE_DATASETS
    }
    root = Path(output_dir).expanduser().resolve() / run_id
    results_file = root / "transfer_results.jsonl"
    if results_file.exists():
        raise FileExistsError(f"Locked-test results already exist: {results_file}")
    root.mkdir(parents=True, exist_ok=True)
    device = cnn.default_device()

    for source_dataset, winner in winners.items():
        print(
            f"[best] {source_dataset} trial #{winner['trial_number']} "
            f"validation_balanced_accuracy={winner['objective']['value']:.6f}"
        )

    for target_dataset in SOURCE_DATASETS:
        calibrations = {}
        for source_dataset, winner in winners.items():
            selected_epochs, fold_results = calibrate_epochs(
                winner["architecture"], target_dataset, seed, device
            )
            calibrations[source_dataset] = {
                "selected_epochs": selected_epochs,
                "fold_results": fold_results,
            }
            print(
                f"[calibrate] selected={source_dataset} target={target_dataset} "
                f"epochs={selected_epochs}"
            )

        loaders, info = make_final_data_loaders(
            target_dataset,
            batch_size=cnn.TRAINING["batch_size"],
            seed=seed,
        )
        for source_dataset, winner in winners.items():
            cnn.seed_everything(seed)
            loaders["train"].generator.manual_seed(seed)
            model = cnn.EEGCNN(info["channels"], winner["architecture"]).to(device)
            parameter_count = sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
            calibration = calibrations[source_dataset]
            final_epochs = calibration["selected_epochs"]
            training_seconds = train_final(
                model, loaders["train"], device, final_epochs
            )
            test_metrics = evaluate(model, loaders["test"], device)

            model_name = f"selected-{source_dataset}_trained-{target_dataset}.pt"
            model_path = root / model_name
            torch.save(
                {
                    "source_dataset": source_dataset,
                    "target_dataset": target_dataset,
                    "source_trial_number": winner["trial_number"],
                    "source_validation_balanced_accuracy": winner["objective"]["value"],
                    "architecture": winner["architecture"],
                    "input_channels": info["channels"],
                    "seed": seed,
                    "epochs": final_epochs,
                    "calibration": calibration,
                    "state_dict": _cpu_state_dict(model),
                },
                model_path,
            )
            entry = {
                "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
                "run_id": run_id,
                "seed": seed,
                "source_dataset": source_dataset,
                "target_dataset": target_dataset,
                "source_trial_number": winner["trial_number"],
                "source_validation_balanced_accuracy": winner["objective"]["value"],
                "architecture": winner["architecture"],
                "training": {
                    **cnn.TRAINING,
                    "calibration_max_epochs": CALIBRATION_MAX_EPOCHS,
                    "calibration_patience": CALIBRATION_PATIENCE,
                    "final_epochs": final_epochs,
                },
                "calibration": calibration,
                "training_seconds": training_seconds,
                "parameter_count": parameter_count,
                "test": test_metrics,
                "model_path": str(model_path),
                "runtime": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "torch": torch.__version__,
                    "numpy": np.__version__,
                    "optuna": optuna.__version__,
                    "device": str(device),
                },
            }
            with results_file.open("a", encoding="utf-8") as file:
                file.write(json.dumps(entry) + "\n")
            print(
                f"[test] selected={source_dataset} trained={target_dataset} "
                f"balanced_accuracy={test_metrics['balanced_accuracy']:.4f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bnci-trials", required=True)
    parser.add_argument("--lee-trials", required=True)
    parser.add_argument("--physionet-trials", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    main(
        trial_files={
            "bnci2014_001": args.bnci_trials,
            "lee2019_mi": args.lee_trials,
            "physionet_mi": args.physionet_trials,
        },
        output_dir=args.output_dir,
        run_id=args.run_id,
        seed=args.seed,
    )
