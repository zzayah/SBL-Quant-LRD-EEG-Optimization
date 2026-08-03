"""EEG CNN blocks, training, and the Optuna classification objective."""

import json
import platform
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import optuna
import torch
from torch import nn


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.eeg_dataset import make_data_loaders
from evaluation import evaluate_model


OUTPUT_DIR = BASE_DIR / "output"
SESSION_ID = "unconfigured"

SEARCH_SPACE = {
    "temporal_filters": [8, 16, 24, 32, 48, 64],
    "temporal_kernel": [15, 31, 63],
    "spatial_depth_multiplier": [1, 2],
    "num_temporal_blocks": [1, 2, 3, 4, 5, 6],
    "block_filters": [16, 24, 32, 48, 64, 96, 128],
    "block_kernel": [7, 15, 31],
    "stem_pool_size": [1, 2, 4],
    "block_pool_size": [1, 2, 4, 8],
    "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
    "activation": ["elu", "gelu", "relu"],
    "head_type": ["flatten", "log_variance"],
}
MAX_PARAMETERS = 350_000

TRAINING = {
    "batch_size": 64,
    "max_epochs": 50,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "early_stopping_patience": 10,
}


def configure_run(output_dir: str | Path, session_id: str) -> None:
    global OUTPUT_DIR, SESSION_ID
    OUTPUT_DIR = Path(output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SESSION_ID = session_id


def log_unsuccessful_trial(
    trial: optuna.trial.Trial,
    dataset_name: str,
    seed: int,
    status: str,
    reason: str,
) -> None:
    entry = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": seed,
        "dataset": dataset_name,
        "trial_number": trial.number,
        "status": status,
        "reason": reason,
        "params": trial.params,
    }
    with (OUTPUT_DIR / f"{SESSION_ID}_{dataset_name}_failures.jsonl").open(
        "a", encoding="utf-8"
    ) as file:
        file.write(json.dumps(entry) + "\n")


def seed_everything(seed: int) -> None:
    """Seed every random source used by the EEG NAS."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def default_device() -> torch.device:
    """Prefer CUDA and otherwise use the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def activation_layer(name: str) -> nn.Module:
    if name == "elu":
        return nn.ELU()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unknown activation: {name}")


class SeparableTemporalBlock(nn.Module):
    """Depthwise temporal filtering followed by channel mixing."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_size: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size,
                padding="same",
                groups=in_channels,
                bias=False,
            ),
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            activation_layer(activation),
            nn.AvgPool1d(pool_size),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class LogVariance(nn.Module):
    """Convert temporal feature maps into log-power features."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.mean(torch.square(inputs), dim=-1).clamp_min(1e-7))


class EEGCNN(nn.Module):
    """Temporal-spatial EEG stem followed by searchable temporal blocks."""

    def __init__(
        self,
        input_channels: int,
        architecture: dict,
        input_samples: int = 384,
    ) -> None:
        super().__init__()
        temporal_filters = architecture["temporal_filters"]
        spatial_filters = temporal_filters * architecture["spatial_depth_multiplier"]
        activation = architecture["activation"]
        dropout = architecture["dropout"]

        self.temporal = nn.Sequential(
            nn.Conv2d(
                1,
                temporal_filters,
                (1, architecture["temporal_kernel"]),
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(temporal_filters),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(
                temporal_filters,
                spatial_filters,
                (input_channels, 1),
                groups=temporal_filters,
                bias=False,
            ),
            nn.BatchNorm2d(spatial_filters),
            activation_layer(activation),
            nn.AvgPool2d((1, architecture["stem_pool_size"])),
            nn.Dropout(dropout),
        )

        blocks = []
        current_channels = spatial_filters
        for block in architecture["temporal_blocks"]:
            blocks.append(
                SeparableTemporalBlock(
                    current_channels,
                    block["out_channels"],
                    block["kernel_size"],
                    block["pool_size"],
                    dropout,
                    activation,
                )
            )
            current_channels = block["out_channels"]
        self.blocks = nn.Sequential(*blocks)
        head_type = architecture.get("head_type", "global_average")
        if head_type == "flatten":
            feature_samples = input_samples // architecture["stem_pool_size"]
            for block in architecture["temporal_blocks"]:
                feature_samples //= block["pool_size"]
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(current_channels * feature_samples, 2),
            )
        elif head_type == "log_variance":
            self.classifier = nn.Sequential(LogVariance(), nn.Linear(current_channels, 2))
        elif head_type == "global_average":
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(current_channels, 2),
            )
        else:
            raise ValueError(f"Unknown head type: {head_type}")

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        features = self.temporal(signals.unsqueeze(1))
        features = self.spatial(features).squeeze(2)
        return self.classifier(self.blocks(features))


def sample_architecture(trial: optuna.trial.Trial) -> dict:
    """Sample one shared EEG architecture from the approved search space."""
    activation = trial.suggest_categorical("activation", SEARCH_SPACE["activation"])
    temporal_filters = trial.suggest_categorical(
        "temporal_filters", SEARCH_SPACE["temporal_filters"]
    )
    architecture = {
        "activation": activation,
        "dropout": trial.suggest_categorical("dropout", SEARCH_SPACE["dropout"]),
        "head_type": trial.suggest_categorical(
            "head_type", SEARCH_SPACE["head_type"]
        ),
        "temporal_filters": temporal_filters,
        "temporal_kernel": trial.suggest_categorical(
            "temporal_kernel", SEARCH_SPACE["temporal_kernel"]
        ),
        "spatial_depth_multiplier": trial.suggest_categorical(
            "spatial_depth_multiplier", SEARCH_SPACE["spatial_depth_multiplier"]
        ),
        "stem_pool_size": trial.suggest_categorical(
            "stem_pool_size", SEARCH_SPACE["stem_pool_size"]
        ),
        "temporal_blocks": [],
    }
    block_count = trial.suggest_categorical(
        "num_temporal_blocks", SEARCH_SPACE["num_temporal_blocks"]
    )
    current_channels = temporal_filters * architecture["spatial_depth_multiplier"]
    feature_samples = 384 // architecture["stem_pool_size"]
    for index in range(block_count):
        filter_rank = trial.suggest_int(
            f"block_{index}_filter_rank",
            0,
            len(SEARCH_SPACE["block_filters"]) - 1,
        )
        proposed_channels = SEARCH_SPACE["block_filters"][filter_rank]
        out_channels = max(current_channels, proposed_channels)
        pool_rank = trial.suggest_int(
            f"block_{index}_pool_rank",
            0,
            len(SEARCH_SPACE["block_pool_size"]) - 1,
        )
        proposed_pool = SEARCH_SPACE["block_pool_size"][pool_rank]
        pool_size = min(proposed_pool, max(feature_samples, 1))
        architecture["temporal_blocks"].append(
            {
                "out_channels": out_channels,
                "kernel_size": trial.suggest_categorical(
                    f"block_{index}_kernel", SEARCH_SPACE["block_kernel"]
                ),
                "pool_size": pool_size,
            }
        )
        current_channels = out_channels
        feature_samples //= pool_size
    return architecture


def train_candidate(
    model: nn.Module,
    train_loader,
    validation_loader,
    device: torch.device,
) -> dict[str, float | int]:
    """Train one fold and return its best validation metrics."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING["learning_rate"],
        weight_decay=TRAINING["weight_decay"],
    )
    best_score = -1.0
    best_metrics = None
    stale_epochs = 0

    for _ in range(TRAINING["max_epochs"]):
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

        metrics = evaluate_model(model, validation_loader, device, criterion)
        if metrics["balanced_accuracy"] > best_score:
            best_score = float(metrics["balanced_accuracy"])
            best_metrics = metrics
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= TRAINING["early_stopping_patience"]:
            break

    if best_metrics is None:
        raise RuntimeError("Candidate training produced no validation result")
    return best_metrics


def objective(
    trial: optuna.trial.Trial,
    dataset_name: str,
    seed: int,
    device: torch.device | None = None,
) -> float:
    """Return mean subject-held-out validation balanced accuracy."""
    try:
        architecture = sample_architecture(trial)
        device = device or default_device()
        folds = range(3) if dataset_name == "bnci2014_001" else [None]
        fold_results = []

        for fold in folds:
            seed_everything(seed)
            loaders, info = make_data_loaders(
                dataset_name,
                batch_size=TRAINING["batch_size"],
                seed=seed,
                fold=fold,
            )
            model = EEGCNN(info["channels"], architecture).to(device)
            parameter_count = sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            )
            if parameter_count > MAX_PARAMETERS:
                raise optuna.TrialPruned(
                    f"{parameter_count:,} parameters exceeds {MAX_PARAMETERS:,}"
                )
            metrics = train_candidate(
                model,
                loaders["train"],
                loaders["validation"],
                device,
            )
            fold_results.append({"fold": fold, **metrics})
    except optuna.TrialPruned as error:
        log_unsuccessful_trial(trial, dataset_name, seed, "pruned", str(error))
        raise
    except (FloatingPointError, RuntimeError) as error:
        log_unsuccessful_trial(trial, dataset_name, seed, "failed", str(error))
        raise

    score = float(np.mean([result["balanced_accuracy"] for result in fold_results]))
    entry = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": seed,
        "dataset": dataset_name,
        "trial_number": trial.number,
        "objective": {"name": "validation_balanced_accuracy", "value": score},
        "params": trial.params,
        "architecture": architecture,
        "training": TRAINING,
        "parameter_count": parameter_count,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "optuna": optuna.__version__,
            "device": str(device),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "fold_results": fold_results,
    }
    with (OUTPUT_DIR / f"{SESSION_ID}_{dataset_name}.jsonl").open(
        "a", encoding="utf-8"
    ) as file:
        file.write(json.dumps(entry) + "\n")
    return score
