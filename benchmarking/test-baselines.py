#!/usr/bin/env python3
"""Retrain calibrated EEG baselines and evaluate locked test subjects."""

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
EEG_PARZEN_DIR = REPO_ROOT / "eeg-parzen"
for path in (REPO_ROOT, EEG_PARZEN_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cnn
from benchmarking.architectures import BASELINES
from benchmarking.evaluation import evaluate
from data.eeg_dataset import SUPPORTED_DATASETS, make_final_data_loaders


def load_results(path: str | Path, expected_dataset: str) -> dict[str, dict]:
    with Path(path).open(encoding="utf-8") as file:
        entries = [json.loads(line) for line in file if line.strip()]
    results = {
        entry["baseline"]: entry
        for entry in entries
        if entry.get("dataset") == expected_dataset
    }
    missing = set(BASELINES) - set(results)
    if missing:
        raise RuntimeError(f"Missing {expected_dataset} baselines: {sorted(missing)}")
    return results


def train_final(model: nn.Module, loader, device: torch.device, epochs: int) -> float:
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
                raise FloatingPointError("Final baseline loss became non-finite")
            loss.backward()
            optimizer.step()
    return time.perf_counter() - started


def main(result_files: dict[str, str], output_dir: str, run_id: str, seed: int) -> None:
    root = Path(output_dir).expanduser().resolve() / run_id
    output_file = root / "baseline_test_results.jsonl"
    if output_file.exists():
        raise FileExistsError(f"Locked-test results already exist: {output_file}")
    root.mkdir(parents=True, exist_ok=True)
    device = cnn.default_device()

    for dataset in SUPPORTED_DATASETS:
        validation_results = load_results(result_files[dataset], dataset)
        loaders, info = make_final_data_loaders(
            dataset,
            batch_size=cnn.TRAINING["batch_size"],
            seed=seed,
        )
        for name, factory in BASELINES.items():
            calibration = validation_results[name]
            epochs = max(
                1,
                int(np.median([fold["best_epoch"] for fold in calibration["fold_results"]])),
            )
            cnn.seed_everything(seed)
            loaders["train"].generator.manual_seed(seed)
            model = factory(info["channels"]).to(device)
            parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            training_seconds = train_final(model, loaders["train"], device, epochs)
            test_metrics = evaluate(model, loaders["test"], device)
            model_path = root / f"{dataset}_{name}.pt"
            torch.save(
                {
                    "dataset": dataset,
                    "baseline": name,
                    "seed": seed,
                    "epochs": epochs,
                    "input_channels": info["channels"],
                    "state_dict": {
                        key: value.detach().cpu() for key, value in model.state_dict().items()
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
                "calibrated_epochs": epochs,
                "calibration": calibration["fold_results"],
                "training_seconds": training_seconds,
                "parameter_count": parameter_count,
                "test": test_metrics,
                "model_path": str(model_path),
                "device": str(device),
            }
            with output_file.open("a", encoding="utf-8") as file:
                file.write(json.dumps(entry) + "\n")
            print(
                f"[test] dataset={dataset} baseline={name} epochs={epochs} "
                f"balanced_accuracy={test_metrics['balanced_accuracy']:.4f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bnci-results", required=True)
    parser.add_argument("--lee-results", required=True)
    parser.add_argument("--physionet-results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    main(
        result_files={
            "bnci2014_001": args.bnci_results,
            "lee2019_mi": args.lee_results,
            "physionet_mi": args.physionet_results,
        },
        output_dir=args.output_dir,
        run_id=args.run_id,
        seed=args.seed,
    )
