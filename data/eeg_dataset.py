"""PyTorch datasets and subject-level loaders for the processed EEG data."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


DATA_ROOT = Path(__file__).resolve().parent
PROCESSED_ROOT = DATA_ROOT / "moabb_processed"
SPLIT_FILE = DATA_ROOT / "seed-0" / "splits_seed0.json"
SUPPORTED_DATASETS = ("bnci2014_001", "lee2019_mi", "physionet_mi")
DATASET_SUBJECTS = {
    "bnci2014_001": list(range(1, 10)),
    "lee2019_mi": list(range(1, 55)),
    "physionet_mi": list(range(1, 110)),
}


def split_file_for_seed(seed: int) -> Path:
    return DATA_ROOT / f"seed-{seed}" / f"splits_seed{seed}.json"


def create_subject_splits(seed: int) -> Path:
    """Write deterministic subject assignments controlled by one seed."""
    rng = np.random.default_rng(seed)
    splits = {"seed": seed, "datasets": {}}
    fixed_counts = {
        "physionet_mi": (70, 17, 22),
        "lee2019_mi": (35, 8, 11),
    }
    for dataset_name, counts in fixed_counts.items():
        shuffled = rng.permutation(DATASET_SUBJECTS[dataset_name]).tolist()
        train_count, validation_count, test_count = counts
        splits["datasets"][dataset_name] = {
            "train": sorted(shuffled[:train_count]),
            "validation": sorted(shuffled[train_count : train_count + validation_count]),
            "test": sorted(shuffled[-test_count:]),
        }

    bnci_subjects = rng.permutation(DATASET_SUBJECTS["bnci2014_001"]).tolist()
    test_subjects = sorted(bnci_subjects[:2])
    development_subjects = bnci_subjects[2:]
    folds = []
    for validation in np.array_split(development_subjects, 3):
        validation_subjects = sorted(validation.tolist())
        folds.append(
            {
                "train": sorted(set(development_subjects) - set(validation_subjects)),
                "validation": validation_subjects,
            }
        )
    splits["datasets"]["bnci2014_001"] = {
        "development": sorted(development_subjects),
        "test": test_subjects,
        "folds": folds,
    }
    output_file = split_file_for_seed(seed)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(splits, file, indent=2)
    return output_file


def load_splits(split_file: Path = SPLIT_FILE) -> dict:
    with split_file.open(encoding="utf-8") as file:
        return json.load(file)


def subject_split(
    dataset_name: str,
    fold: int | None = None,
    split_file: Path = SPLIT_FILE,
) -> dict[str, list[int]]:
    """Return the train, validation, and locked test subject IDs."""
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    split = load_splits(split_file)["datasets"][dataset_name]
    if dataset_name == "bnci2014_001":
        if fold is None or fold not in range(len(split["folds"])):
            raise ValueError("BNCI2014_001 requires fold=0, fold=1, or fold=2")
        return {
            "train": split["folds"][fold]["train"],
            "validation": split["folds"][fold]["validation"],
            "test": split["test"],
        }
    if fold is not None:
        raise ValueError(f"{dataset_name} does not use cross-validation folds")
    return {key: split[key] for key in ("train", "validation", "test")}


def subject_file(dataset_name: str, subject: int, processed_root: Path) -> Path:
    return processed_root / dataset_name / f"subject_{subject:03d}.npz"


def compute_channel_stats(
    dataset_name: str,
    subjects: list[int],
    processed_root: Path = PROCESSED_ROOT,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean and standard deviation from training subjects."""
    channel_sum = None
    channel_square_sum = None
    sample_count = 0
    expected_channels = None

    for subject in subjects:
        path = subject_file(dataset_name, subject, processed_root)
        with np.load(path) as data:
            signals = data["X"]
            channels = tuple(data["channels"].tolist())
            if expected_channels is None:
                expected_channels = channels
                channel_sum = np.zeros(signals.shape[1], dtype=np.float64)
                channel_square_sum = np.zeros(signals.shape[1], dtype=np.float64)
            elif channels != expected_channels:
                raise ValueError(f"Channel order differs in {path}")
            channel_sum += signals.sum(axis=(0, 2), dtype=np.float64)
            channel_square_sum += np.square(signals, dtype=np.float64).sum(axis=(0, 2))
            sample_count += signals.shape[0] * signals.shape[2]

    if sample_count == 0 or channel_sum is None or channel_square_sum is None:
        raise ValueError("At least one training subject is required")
    mean = channel_sum / sample_count
    variance = np.maximum(channel_square_sum / sample_count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    if np.any(std == 0):
        raise ValueError(f"Zero-variance channel found in {dataset_name}")
    return mean.astype(np.float32), std.astype(np.float32)


class SubjectEEGDataset(Dataset):
    """In-memory EEG trials from a collection of complete subjects."""

    def __init__(
        self,
        dataset_name: str,
        subjects: list[int],
        channel_mean: np.ndarray,
        channel_std: np.ndarray,
        processed_root: Path = PROCESSED_ROOT,
    ) -> None:
        signals = []
        targets = []
        trial_subjects = []
        channel_names = None

        for subject in subjects:
            path = subject_file(dataset_name, subject, processed_root)
            with np.load(path) as data:
                subject_signals = data["X"].astype(np.float32, copy=True)
                subject_targets = data["y"].astype(np.int64, copy=True)
                subject_channels = tuple(data["channels"].tolist())
            if channel_names is None:
                channel_names = subject_channels
            elif subject_channels != channel_names:
                raise ValueError(f"Channel order differs in {path}")
            signals.append(subject_signals)
            targets.append(subject_targets)
            trial_subjects.extend([subject] * len(subject_targets))

        if not signals:
            raise ValueError("At least one subject is required")
        combined_signals = np.concatenate(signals)
        combined_signals -= channel_mean[None, :, None]
        combined_signals /= channel_std[None, :, None]

        self.signals = torch.from_numpy(combined_signals)
        self.targets = torch.from_numpy(np.concatenate(targets))
        self.trial_subjects = np.asarray(trial_subjects, dtype=np.int64)
        self.subjects = tuple(subjects)
        self.channel_names = channel_names

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.signals[index], self.targets[index]


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_data_loaders(
    dataset_name: str,
    batch_size: int,
    seed: int,
    fold: int | None = None,
    include_test: bool = False,
    num_workers: int = 0,
    processed_root: Path = PROCESSED_ROOT,
    split_file: Path | None = None,
) -> tuple[dict[str, DataLoader], dict]:
    """Build deterministic subject-held-out loaders and training-only stats."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    split_file = split_file or split_file_for_seed(seed)
    subjects = subject_split(dataset_name, fold=fold, split_file=split_file)
    mean, std = compute_channel_stats(dataset_name, subjects["train"], processed_root)
    partitions = ["train", "validation"]
    if include_test:
        partitions.append("test")

    datasets = {
        name: SubjectEEGDataset(
            dataset_name,
            subjects[name],
            mean,
            std,
            processed_root,
        )
        for name in partitions
    }
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=name == "train",
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=generator,
            pin_memory=torch.cuda.is_available(),
        )
        for name, dataset in datasets.items()
    }
    info = {
        "dataset": dataset_name,
        "fold": fold,
        "subjects": {name: subjects[name] for name in partitions},
        "channel_mean": mean,
        "channel_std": std,
        "channels": len(next(iter(datasets.values())).channel_names),
        "samples": next(iter(datasets.values())).signals.shape[-1],
        "classes": 2,
    }
    return loaders, info


def make_final_data_loaders(
    dataset_name: str,
    batch_size: int,
    seed: int,
    include_test: bool = True,
    num_workers: int = 0,
    processed_root: Path = PROCESSED_ROOT,
    split_file: Path | None = None,
) -> tuple[dict[str, DataLoader], dict]:
    """Build a combined-development loader and optionally the locked-test loader."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    split_file = split_file or split_file_for_seed(seed)
    split = load_splits(split_file)["datasets"][dataset_name]
    if dataset_name == "bnci2014_001":
        train_subjects = split["development"]
    else:
        train_subjects = sorted(split["train"] + split["validation"])
    subjects = {"train": train_subjects}
    if include_test:
        subjects["test"] = split["test"]
    mean, std = compute_channel_stats(dataset_name, train_subjects, processed_root)
    datasets = {
        name: SubjectEEGDataset(
            dataset_name,
            partition_subjects,
            mean,
            std,
            processed_root,
        )
        for name, partition_subjects in subjects.items()
    }
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=name == "train",
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=generator,
            pin_memory=torch.cuda.is_available(),
        )
        for name, dataset in datasets.items()
    }
    info = {
        "dataset": dataset_name,
        "subjects": subjects,
        "channel_mean": mean,
        "channel_std": std,
        "channels": len(datasets["train"].channel_names),
        "samples": datasets["train"].signals.shape[-1],
        "classes": 2,
    }
    return loaders, info
