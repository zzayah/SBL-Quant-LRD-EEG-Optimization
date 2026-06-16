import importlib.util
import json
from pathlib import Path

import numpy as np
import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import trange


DATA_PATH = Path("data/driver-drowsiness/orosco/dataset/orosco_windowed_balanced.npz")
OUT_PATH = Path("models/optim/quantization/cnn-trials.jsonl")
N_TRIALS = 25
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-4
DEVICE = "mps"
PARAM_PENALTY_PER_100K = 0.01


def load_optimization_report():
    metrics_path = Path(__file__).with_name("basic-nas-optim-param.py")
    spec = importlib.util.spec_from_file_location("basic_nas_optim_param", metrics_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load optimization metrics from {metrics_path}")
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    return metrics.optimization_report


optimization_report = load_optimization_report()

data = np.load(DATA_PATH)
x = torch.tensor(data["X"], dtype=torch.float32)
y = torch.tensor(data["Y"], dtype=torch.long)
n_train = int(0.8 * len(x))
train_loader = DataLoader(TensorDataset(x[:n_train], y[:n_train]), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(x[n_train:], y[n_train:]), batch_size=BATCH_SIZE)
num_classes = int(y.max()) + 1


class DynamicCNN:
    def __init__(self, trial: optuna.trial.Trial, in_channels: int, length: int, num_classes: int) -> None:
        self.activation_name = trial.suggest_categorical("activation", ["relu", "elu"])
        self.num_blocks = trial.suggest_categorical("num_blocks", [1, 2, 3])
        fc_options = {"fc16": [16], "fc32": [32], "fc64": [64], "fc128": [128], "fc64_32": [64, 32]}
        self.fc_layers = fc_options[trial.suggest_categorical("fc_layers", list(fc_options))]

        self.conv_layers = []
        for i in range(self.num_blocks):
            out_channels = trial.suggest_categorical(f"conv_{i}_channels", [4, 8, 16, 24, 32])
            self.conv_layers.append((in_channels, out_channels, 3, 1, 0))
            in_channels = out_channels
            length -= 2

        activation = nn.ReLU if self.activation_name == "relu" else nn.ELU
        layers = []
        for conv_in_channels, out_channels, kernel_size, stride, padding in self.conv_layers:
            layers += [nn.Conv1d(conv_in_channels, out_channels, kernel_size, stride, padding), activation()]

        layers.append(nn.Flatten())
        in_features = in_channels * length
        for out_features in self.fc_layers:
            layers += [nn.Linear(in_features, out_features), activation()]
            in_features = out_features
        layers.append(nn.Linear(in_features, num_classes))

        self.model = nn.Sequential(*layers)

    def spec(self) -> dict:
        return {
            "activation": self.activation_name,
            "conv_layers": self.conv_layers,
            "fc_layers": self.fc_layers,
            "nn.Sequential": repr(self.model),
        }


def accuracy(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += len(yb)
    return correct / total


def objective(trial):
    cnn = DynamicCNN(trial, x.shape[1], x.shape[2], num_classes)
    model = cnn.model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    epoch_bar = trange(EPOCHS, desc=f"Trial {trial.number}", leave=False)
    for _ in epoch_bar:
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_bar.set_postfix(loss=f"{loss.item():.4f}")

    train_acc = accuracy(model, train_loader)
    val_acc = accuracy(model, val_loader)
    optimization = optimization_report(val_acc, model, PARAM_PENALTY_PER_100K)
    with OUT_PATH.open("a") as f:
        f.write(json.dumps({
            "trial": trial.number,
            "accuracies": {"train": train_acc, "validation": val_acc},
            "optimization": optimization,
            "params": trial.params,
            "training": {"optimizer": "adam", "lr": LR, "batch_size": BATCH_SIZE, "epochs": EPOCHS},
            "architecture": cnn.spec(),
        }) + "\n")
    return optimization["score"]
