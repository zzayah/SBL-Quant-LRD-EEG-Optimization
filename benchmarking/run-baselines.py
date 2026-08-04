#!/usr/bin/env python3
"""Train fixed EEG baselines on the same validation splits as the NAS."""

import argparse
import copy
import importlib.util
import json
import secrets
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
from data.eeg_dataset import SUPPORTED_DATASETS, make_data_loaders


def _control_seed() -> int:
    path = EEG_PARZEN_DIR / "control-logic.py"
    spec = importlib.util.spec_from_file_location("eeg_control_logic", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SEED


SEED = _control_seed()
CALIBRATION_MAX_EPOCHS = 100
CALIBRATION_PATIENCE = 15


def train(
    model: nn.Module,
    train_loader,
    validation_loader,
    device: torch.device,
) -> tuple[dict, int, int, float]:
    """Train with the NAS constants and restore the best validation weights."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cnn.TRAINING["learning_rate"],
        weight_decay=cnn.TRAINING["weight_decay"],
    )
    best_score = -1.0
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    started = time.perf_counter()

    for epoch in range(1, CALIBRATION_MAX_EPOCHS + 1):
        model.train()
        for signals, targets in train_loader:
            signals = signals.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(signals), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss became non-finite")
            loss.backward()
            optimizer.step()

        metrics = cnn.evaluate_model(model, validation_loader, device, criterion)
        if metrics["balanced_accuracy"] > best_score:
            best_score = metrics["balanced_accuracy"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= CALIBRATION_PATIENCE:
            break

    if best_state is None:
        raise RuntimeError("Baseline training produced no validation result")
    model.load_state_dict(best_state)
    training_seconds = time.perf_counter() - started
    return evaluate(model, validation_loader, device), best_epoch, epoch, training_seconds


def run_dataset(
    dataset: str,
    baseline_names: list[str],
    output_root: Path,
    session_id: str,
) -> None:
    output_dir = output_root / dataset / session_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "baseline_results.jsonl"
    completed = set()
    if output_file.exists():
        with output_file.open(encoding="utf-8") as file:
            completed = {
                json.loads(line)["baseline"] for line in file if line.strip()
            }
    folds = range(3) if dataset == "bnci2014_001" else [None]
    device = cnn.default_device()

    for name in baseline_names:
        factory = BASELINES[name]
        if name in completed:
            print(f"{dataset} {name}: already complete")
            continue
        fold_results = []
        parameter_count = None
        model_description = None
        for fold in folds:
            cnn.seed_everything(SEED)
            loaders, info = make_data_loaders(
                dataset,
                batch_size=cnn.TRAINING["batch_size"],
                seed=SEED,
                fold=fold,
            )
            model = factory(info["channels"]).to(device)
            parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model_description = repr(model)
            metrics, best_epoch, epochs_run, training_seconds = train(
                model, loaders["train"], loaders["validation"], device
            )
            fold_results.append(
                {
                    "fold": fold,
                    "best_epoch": best_epoch,
                    "epochs_run": epochs_run,
                    "training_seconds": training_seconds,
                    **metrics,
                }
            )

        aggregate_keys = (
            "loss",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "left_recall",
            "right_recall",
            "left_precision",
            "right_precision",
            "inference_ms_per_trial",
            "inference_trials_per_second",
        )
        entry = {
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "session_id": session_id,
            "seed": SEED,
            "dataset": dataset,
            "baseline": name,
            "device": str(device),
            "parameter_count": parameter_count,
            "model": model_description,
            "training": {
                **cnn.TRAINING,
                "calibration_max_epochs": CALIBRATION_MAX_EPOCHS,
                "calibration_patience": CALIBRATION_PATIENCE,
            },
            "validation": {
                key: float(np.mean([result[key] for result in fold_results]))
                for key in aggregate_keys
            },
            "fold_results": fold_results,
        }
        with output_file.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry) + "\n")
        print(
            f"{dataset} {name}: balanced_accuracy="
            f"{entry['validation']['balanced_accuracy']:.4f}"
        )


def main(
    datasets: list[str],
    baseline_names: list[str],
    output_dir: str,
    session_id: str,
) -> None:
    for dataset in datasets:
        run_dataset(dataset, baseline_names, Path(output_dir), session_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("all", *SUPPORTED_DATASETS),
        default="all",
    )
    parser.add_argument(
        "--output-dir",
        default=f"data/seed-{SEED}/benchmarking",
    )
    parser.add_argument(
        "--baseline",
        nargs="+",
        choices=tuple(BASELINES),
        default=list(BASELINES),
    )
    parser.add_argument("--session-id", default=secrets.token_hex(4))
    args = parser.parse_args()
    selected = list(SUPPORTED_DATASETS) if args.dataset == "all" else [args.dataset]
    print(f"Run ID: {args.session_id}")
    main(selected, args.baseline, args.output_dir, args.session_id)
