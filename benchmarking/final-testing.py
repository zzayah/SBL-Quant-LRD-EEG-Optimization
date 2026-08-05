#!/usr/bin/env python3
"""Evaluate frozen final baseline models on the locked test subjects."""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
EEG_PARZEN_DIR = REPO_ROOT / "eeg-parzen"
for path in (REPO_ROOT, EEG_PARZEN_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cnn
from benchmarking.architectures import BASELINES
from benchmarking.evaluation import evaluate
from data.eeg_dataset import SUPPORTED_DATASETS, make_final_data_loaders


def main(model_dir: str, output_dir: str, run_id: str, seed: int, epochs: int) -> None:
    model_root = Path(model_dir).expanduser().resolve()
    root = Path(output_dir).expanduser().resolve() / run_id / f"epochs-{epochs}"
    results_file = root / "test_results.jsonl"
    if results_file.exists():
        raise FileExistsError(f"Locked-test results already exist: {results_file}")
    root.mkdir(parents=True, exist_ok=True)
    device = cnn.default_device()

    for dataset in SUPPORTED_DATASETS:
        loaders, info = make_final_data_loaders(
            dataset,
            batch_size=cnn.TRAINING["batch_size"],
            seed=seed,
        )
        for name, factory in BASELINES.items():
            model_path = model_root / f"{name}_trained-{dataset}.pt"
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            if checkpoint["seed"] != seed or checkpoint["epochs"] != epochs:
                raise RuntimeError(f"Unexpected checkpoint metadata in {model_path}")
            model = factory(info["channels"]).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            test_metrics = evaluate(model, loaders["test"], device)
            entry = {
                "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
                "run_id": run_id,
                "seed": seed,
                "dataset": dataset,
                "baseline": name,
                "epochs": epochs,
                "test": test_metrics,
                "model_path": str(model_path),
                "device": str(device),
            }
            with results_file.open("a", encoding="utf-8") as file:
                file.write(json.dumps(entry) + "\n")
            print(
                f"[test] baseline={name} target={dataset} "
                f"balanced_accuracy={test_metrics['balanced_accuracy']:.4f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    args = parser.parse_args()
    main(args.model_dir, args.output_dir, args.run_id, args.seed, args.epochs)
