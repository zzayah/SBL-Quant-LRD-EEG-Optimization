#!/usr/bin/env python3
"""Entry point for the subject-independent EEG architecture searches."""

import argparse
import os
import sys


SEED = int(os.environ.get("EEG_SEED", "0"))

if __name__ == "__main__" and "--print-seed" in sys.argv:
    print(SEED)
    raise SystemExit(0)

import optuna

import cnn
from data.eeg_dataset import SUPPORTED_DATASETS, create_subject_splits

DATASETS = tuple(SUPPORTED_DATASETS)
N_TRIALS = 100


def _run_search(dataset: str, output_dir: str, session_id: str, seed: int) -> None:
    """Run one dataset-specific TPE architecture study."""
    dataset_output = f"{output_dir}/{dataset}/{session_id}"
    cnn.configure_run(dataset_output, session_id)
    sampler = optuna.samplers.TPESampler(seed=seed)
    storage = f"sqlite:///{cnn.OUTPUT_DIR.resolve() / 'study.sqlite3'}"
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        storage=storage,
        study_name=f"{session_id}_{dataset}",
        load_if_exists=True,
    )
    remaining_trials = max(0, N_TRIALS - len(study.trials))
    if remaining_trials == 0:
        print(f"{dataset}: study already contains {N_TRIALS} trials.")
        return
    study.optimize(
        lambda trial: cnn.objective(trial, dataset_name=dataset, seed=seed),
        n_trials=remaining_trials,
        catch=(FloatingPointError, RuntimeError),
    )


def main(
    output_dir: str,
    session_id: str,
    seed: int = SEED,
    datasets: tuple[str, ...] = DATASETS,
) -> None:
    """Create the seeded splits and run the requested dataset studies."""
    create_subject_splits(seed)
    for dataset in datasets:
        _run_search(dataset, output_dir, session_id, seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-seed", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--session-id")
    parser.add_argument(
        "--dataset",
        choices=("all", *DATASETS),
        default="all",
    )
    args = parser.parse_args()
    if not all((args.output_dir, args.session_id)):
        parser.error("--output-dir and --session-id are required")
    selected = DATASETS if args.dataset == "all" else (args.dataset,)
    main(args.output_dir, args.session_id, SEED, selected)
