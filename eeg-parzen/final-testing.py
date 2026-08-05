#!/usr/bin/env python3
"""Evaluate frozen final NAS transfer models on the locked test subjects."""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
for path in (REPO_ROOT, BASE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cnn
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

    for target_dataset in SUPPORTED_DATASETS:
        loaders, info = make_final_data_loaders(
            target_dataset,
            batch_size=cnn.TRAINING["batch_size"],
            seed=seed,
        )
        for source_dataset in SUPPORTED_DATASETS:
            model_name = f"selected-{source_dataset}_trained-{target_dataset}.pt"
            model_path = model_root / model_name
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            if checkpoint["seed"] != seed or checkpoint["epochs"] != epochs:
                raise RuntimeError(f"Unexpected checkpoint metadata in {model_path}")
            model = cnn.EEGCNN(info["channels"], checkpoint["architecture"]).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            test_metrics = evaluate(model, loaders["test"], device)
            entry = {
                "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
                "run_id": run_id,
                "seed": seed,
                "source_dataset": source_dataset,
                "target_dataset": target_dataset,
                "source_trial_number": checkpoint["source_trial_number"],
                "epochs": epochs,
                "test": test_metrics,
                "model_path": str(model_path),
                "device": str(device),
            }
            with results_file.open("a", encoding="utf-8") as file:
                file.write(json.dumps(entry) + "\n")
            print(
                f"[test] selected={source_dataset} target={target_dataset} "
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
