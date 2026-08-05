#!/usr/bin/env python3
"""Train the NAS transfer architectures on all development subjects."""

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
for path in (REPO_ROOT, BASE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cnn
from data.eeg_dataset import SUPPORTED_DATASETS, make_final_data_loaders


EXPECTED_TRIALS = 100


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


def train(model, loader, device, epochs: int) -> float:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cnn.TRAINING["learning_rate"],
        weight_decay=cnn.TRAINING["weight_decay"],
    )
    started = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for signals, targets in loader:
            signals = signals.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(signals), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Final training loss became non-finite")
            loss.backward()
            optimizer.step()
    return time.perf_counter() - started


def main(
    trial_files: dict[str, str],
    output_dir: str,
    run_id: str,
    seed: int,
    epochs: int,
) -> None:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    winners = {
        dataset: load_best(trial_files[dataset], dataset)
        for dataset in SUPPORTED_DATASETS
    }
    root = Path(output_dir).expanduser().resolve() / run_id / f"epochs-{epochs}"
    results_file = root / "training_results.jsonl"
    if results_file.exists():
        raise FileExistsError(f"Final training results already exist: {results_file}")
    root.mkdir(parents=True, exist_ok=True)
    device = cnn.default_device()

    for target_dataset in SUPPORTED_DATASETS:
        loaders, info = make_final_data_loaders(
            target_dataset,
            batch_size=cnn.TRAINING["batch_size"],
            seed=seed,
            include_test=False,
        )
        for source_dataset, winner in winners.items():
            cnn.seed_everything(seed)
            loaders["train"].generator.manual_seed(seed)
            model = cnn.EEGCNN(info["channels"], winner["architecture"]).to(device)
            training_seconds = train(model, loaders["train"], device, epochs)
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
                    "epochs": epochs,
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
                "epochs": epochs,
                "training_seconds": training_seconds,
                "parameter_count": sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
                "model_path": str(model_path),
            }
            with results_file.open("a", encoding="utf-8") as file:
                file.write(json.dumps(entry) + "\n")
            print(
                f"[train] selected={source_dataset} target={target_dataset} "
                f"epochs={epochs}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bnci-trials", required=True)
    parser.add_argument("--lee-trials", required=True)
    parser.add_argument("--physionet-trials", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
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
        epochs=args.epochs,
    )
