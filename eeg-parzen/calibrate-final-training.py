#!/usr/bin/env python3
"""Measure 10-epoch learning curves for the NAS transfer architectures."""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
for path in (REPO_ROOT, BASE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cnn
from data.eeg_dataset import SUPPORTED_DATASETS, make_data_loaders


EXPECTED_TRIALS = 100
MAX_EPOCHS = 100
CHECKPOINT_INTERVAL = 10


def load_best(path: str | Path, expected_dataset: str) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        entries = [json.loads(line) for line in file if line.strip()]
    entries = [entry for entry in entries if entry.get("dataset") == expected_dataset]
    if len(entries) != EXPECTED_TRIALS:
        raise RuntimeError(
            f"Expected {EXPECTED_TRIALS} completed {expected_dataset} trials in "
            f"{path}, found {len(entries)}"
        )
    return max(entries, key=lambda entry: float(entry["objective"]["value"]))


def train_epoch(model, loader, device, criterion, optimizer) -> None:
    model.train()
    for signals, targets in loader:
        signals = signals.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(signals)
        loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Calibration training loss became non-finite")
        loss.backward()
        optimizer.step()


def evaluate_curve(model, loader, device) -> dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    samples = 0
    confusion = torch.zeros((2, 2), dtype=torch.int64)
    with torch.no_grad():
        for signals, targets in loader:
            signals = signals.to(device)
            targets = targets.to(device)
            logits = model(signals)
            loss_sum += float(criterion(logits, targets))
            samples += len(targets)
            predictions = logits.argmax(1)
            for target, prediction in zip(
                targets.cpu(), predictions.cpu(), strict=True
            ):
                confusion[target, prediction] += 1
    recalls = confusion.diag() / confusion.sum(1).clamp_min(1)
    return {
        "loss": loss_sum / samples,
        "accuracy": float(confusion.diag().sum() / samples),
        "balanced_accuracy": float(recalls.mean()),
    }


def main(trial_files: dict[str, str], output_dir: str, run_id: str, seed: int) -> None:
    winners = {
        dataset: load_best(trial_files[dataset], dataset)
        for dataset in SUPPORTED_DATASETS
    }
    root = Path(output_dir).expanduser().resolve() / run_id
    results_file = root / "calibration_results.jsonl"
    if results_file.exists():
        raise FileExistsError(f"Calibration results already exist: {results_file}")
    root.mkdir(parents=True, exist_ok=True)
    device = cnn.default_device()

    for target_dataset in SUPPORTED_DATASETS:
        fold = 0 if target_dataset == "bnci2014_001" else None
        loaders, info = make_data_loaders(
            target_dataset,
            batch_size=cnn.TRAINING["batch_size"],
            seed=seed,
            fold=fold,
        )
        train_evaluation_loader = DataLoader(
            loaders["train"].dataset,
            batch_size=cnn.TRAINING["batch_size"],
            shuffle=False,
        )
        for source_dataset, winner in winners.items():
            cnn.seed_everything(seed)
            loaders["train"].generator.manual_seed(seed)
            model = cnn.EEGCNN(info["channels"], winner["architecture"]).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cnn.TRAINING["learning_rate"],
                weight_decay=cnn.TRAINING["weight_decay"],
            )
            for epoch in range(1, MAX_EPOCHS + 1):
                train_epoch(model, loaders["train"], device, criterion, optimizer)
                if epoch % CHECKPOINT_INTERVAL:
                    continue
                train_metrics = evaluate_curve(model, train_evaluation_loader, device)
                validation_metrics = evaluate_curve(
                    model, loaders["validation"], device
                )
                model_name = (
                    f"selected-{source_dataset}_trained-{target_dataset}"
                    f"_epoch-{epoch}.pt"
                )
                model_path = root / model_name
                torch.save(
                    {
                        "source_dataset": source_dataset,
                        "target_dataset": target_dataset,
                        "source_trial_number": winner["trial_number"],
                        "architecture": winner["architecture"],
                        "input_channels": info["channels"],
                        "seed": seed,
                        "epoch": epoch,
                        "state_dict": {
                            name: value.detach().cpu()
                            for name, value in model.state_dict().items()
                        },
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
                    "epoch": epoch,
                    "train": train_metrics,
                    "validation": validation_metrics,
                    "model_path": str(model_path),
                }
                with results_file.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(entry) + "\n")
                print(
                    f"[calibrate] selected={source_dataset} target={target_dataset} "
                    f"epoch={epoch} train={train_metrics['balanced_accuracy']:.4f} "
                    f"validation={validation_metrics['balanced_accuracy']:.4f}"
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
