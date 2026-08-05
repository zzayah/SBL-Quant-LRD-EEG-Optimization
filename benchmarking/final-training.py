#!/usr/bin/env python3
"""Train fixed EEG baselines on all development subjects."""

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
EEG_PARZEN_DIR = REPO_ROOT / "eeg-parzen"
for path in (REPO_ROOT, EEG_PARZEN_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cnn
from benchmarking.architectures import BASELINES
from data.eeg_dataset import SUPPORTED_DATASETS, make_final_data_loaders


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
                raise FloatingPointError("Final baseline training loss became non-finite")
            loss.backward()
            optimizer.step()
    return time.perf_counter() - started


def main(output_dir: str, run_id: str, seed: int, epochs: int) -> None:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    root = Path(output_dir).expanduser().resolve() / run_id / f"epochs-{epochs}"
    results_file = root / "training_results.jsonl"
    if results_file.exists():
        raise FileExistsError(f"Final training results already exist: {results_file}")
    root.mkdir(parents=True, exist_ok=True)
    device = cnn.default_device()

    for dataset in SUPPORTED_DATASETS:
        loaders, info = make_final_data_loaders(
            dataset,
            batch_size=cnn.TRAINING["batch_size"],
            seed=seed,
            include_test=False,
        )
        for name, factory in BASELINES.items():
            cnn.seed_everything(seed)
            loaders["train"].generator.manual_seed(seed)
            model = factory(info["channels"]).to(device)
            training_seconds = train(model, loaders["train"], device, epochs)
            model_path = root / f"{name}_trained-{dataset}.pt"
            torch.save(
                {
                    "dataset": dataset,
                    "baseline": name,
                    "input_channels": info["channels"],
                    "seed": seed,
                    "epochs": epochs,
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
            print(f"[train] baseline={name} target={dataset} epochs={epochs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    args = parser.parse_args()
    main(args.output_dir, args.run_id, args.seed, args.epochs)
