#!/usr/bin/env python3
"""Measure 10-epoch learning curves for the fixed EEG baselines."""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
EEG_PARZEN_DIR = REPO_ROOT / "eeg-parzen"
for path in (REPO_ROOT, EEG_PARZEN_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cnn
from benchmarking.architectures import BASELINE_SUITES
from data.eeg_dataset import SUPPORTED_DATASETS, make_data_loaders


MAX_EPOCHS = 100
CHECKPOINT_INTERVAL = 10


def train_epoch(model, loader, device, criterion, optimizer) -> None:
    model.train()
    for signals, targets in loader:
        signals = signals.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(signals), targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Baseline calibration loss became non-finite")
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
            for target, prediction in zip(targets.cpu(), predictions.cpu(), strict=True):
                confusion[target, prediction] += 1
    recalls = confusion.diag() / confusion.sum(1).clamp_min(1)
    return {
        "loss": loss_sum / samples,
        "accuracy": float(confusion.diag().sum() / samples),
        "balanced_accuracy": float(recalls.mean()),
    }


def main(output_dir: str, run_id: str, seed: int, suite: str) -> None:
    root = Path(output_dir).expanduser().resolve() / run_id
    results_file = root / "calibration_results.jsonl"
    if results_file.exists():
        raise FileExistsError(f"Calibration results already exist: {results_file}")
    root.mkdir(parents=True, exist_ok=True)
    device = cnn.default_device()

    for dataset in SUPPORTED_DATASETS:
        fold = 0 if dataset == "bnci2014_001" else None
        loaders, info = make_data_loaders(
            dataset,
            batch_size=cnn.TRAINING["batch_size"],
            seed=seed,
            fold=fold,
        )
        train_evaluation_loader = DataLoader(
            loaders["train"].dataset,
            batch_size=cnn.TRAINING["batch_size"],
            shuffle=False,
        )
        for name, factory in BASELINE_SUITES[suite].items():
            cnn.seed_everything(seed)
            loaders["train"].generator.manual_seed(seed)
            model = factory(info["channels"]).to(device)
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
                validation_metrics = evaluate_curve(model, loaders["validation"], device)
                model_path = root / f"{name}_trained-{dataset}_epoch-{epoch}.pt"
                torch.save(
                    {
                        "dataset": dataset,
                        "baseline": name,
                        "input_channels": info["channels"],
                        "seed": seed,
                        "epoch": epoch,
                        "state_dict": {
                            key: value.detach().cpu()
                            for key, value in model.state_dict().items()
                        },
                    },
                    model_path,
                )
                entry = {
                    "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
                    "run_id": run_id,
                    "seed": seed,
                    "dataset": dataset,
                    "baseline": name,
                    "epoch": epoch,
                    "train": train_metrics,
                    "validation": validation_metrics,
                    "model_path": str(model_path),
                }
                with results_file.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(entry) + "\n")
                print(
                    f"[calibrate] baseline={name} target={dataset} epoch={epoch} "
                    f"train={train_metrics['balanced_accuracy']:.4f} "
                    f"validation={validation_metrics['balanced_accuracy']:.4f}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--suite", choices=tuple(BASELINE_SUITES), default="fixed")
    args = parser.parse_args()
    main(args.output_dir, args.run_id, args.seed, args.suite)
