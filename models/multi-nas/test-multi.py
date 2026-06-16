import json
from pathlib import Path

import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import trange

from multi import BATCH_SIZE, EPOCHS, LR, TRIALS_PATH, DynamicMultiNet, loss_for, num_classes, x, y


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS_TO_TRAIN = EPOCHS


def load_best_trial(path: Path) -> dict:
    best_trial = None
    best_accuracy = None

    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            trial = json.loads(line)
            try:
                validation_accuracy = trial["accuracies"]["validation"]
            except KeyError as error:
                raise KeyError(f"{path}:{line_number} is missing {error}") from error

            if best_accuracy is None or validation_accuracy > best_accuracy:
                best_trial = trial
                best_accuracy = validation_accuracy

    if best_trial is None:
        raise ValueError(f"No trials found in {path}")

    return best_trial


def train(model: nn.Module, loader: DataLoader, epochs: int, lr: float) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epoch_bar = trange(epochs, desc="Retrain", leave=False)
    for _ in epoch_bar:
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_for(model, model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_bar.set_postfix(loss=f"{loss.item():.4f}")


def evaluate_accuracy(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += len(yb)
    return correct / total


def main() -> None:
    best_trial = load_best_trial(TRIALS_PATH)
    training = best_trial.get("training", {})
    batch_size = int(training.get("batch_size", BATCH_SIZE))
    lr = float(training.get("lr", LR))

    n_train = int(0.8 * len(x))
    train_loader = DataLoader(TensorDataset(x[:n_train], y[:n_train]), batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(TensorDataset(x[n_train:], y[n_train:]), batch_size=batch_size)

    fixed_trial = optuna.trial.FixedTrial(best_trial["params"])
    model = DynamicMultiNet(fixed_trial, x.shape[1], x.shape[2], num_classes).to(DEVICE)

    train(model, train_loader, EPOCHS_TO_TRAIN, lr)
    validation_accuracy = evaluate_accuracy(model, validation_loader)
    print(validation_accuracy)


if __name__ == "__main__":
    main()
