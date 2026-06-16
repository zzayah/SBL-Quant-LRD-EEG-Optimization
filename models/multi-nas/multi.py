import json
from pathlib import Path

import numpy as np
import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import trange


DATA_PATH = Path("data/driver-drowsiness/orosco/dataset/orosco_windowed_balanced.npz")
TRIALS_PATH = Path("models/multi-nas/multi-trials.jsonl")
N_TRIALS = 25
BATCH_SIZE = 32
EPOCHS = 25
LR = 1e-4
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


data = np.load(DATA_PATH)
x = torch.tensor(data["X"], dtype=torch.float32)
y = torch.tensor(data["Y"], dtype=torch.long)
n_train = int(0.8 * len(x))
train_loader = DataLoader(TensorDataset(x[:n_train], y[:n_train]), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(x[n_train:], y[n_train:]), batch_size=BATCH_SIZE)
num_classes = int(y.max()) + 1


class ResidualMLPBlock(nn.Module):
    def __init__(self, features: int, hidden: int, activation: type[nn.Module]) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(features, hidden),
            activation(),
            nn.Linear(hidden, features),
            activation(),
        )

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        return xb + self.layers(xb)


class DeepBoltzmannBlock(nn.Module):
    def __init__(self, layer_sizes: list[int]) -> None:
        super().__init__()
        layers = []
        for in_features, out_features in zip(layer_sizes, layer_sizes[1:]):
            layers.extend([nn.Linear(in_features, out_features), nn.Sigmoid()])
        self.layers = nn.Sequential(*layers)

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        return self.layers(xb)


class CNNBlock(nn.Module):
    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        if xb.ndim == 2:
            xb = xb.unsqueeze(1)
        return self.block(xb).flatten(1)


class DynamicMultiNet(nn.Module):
    def __init__(self, trial: optuna.trial.Trial, in_channels: int, length: int, num_classes: int) -> None:
        super().__init__()
        self.block_specs = []
        self.input_channels = in_channels
        self.input_length = length
        self.activation_name = trial.suggest_categorical("activation", ["relu", "elu"])
        self.num_blocks = trial.suggest_categorical("num_blocks", [1, 2, 3])

        activation = nn.ReLU if self.activation_name == "relu" else nn.ELU
        self.blocks = nn.ModuleList()
        features = in_channels
        vector_features = in_channels * length

        for block_index in range(self.num_blocks):
            block_type = trial.suggest_categorical(f"block_{block_index}_type", ["cnn", "mlp", "rnn", "dbm"])
            if block_type == "rnn":
                structure = self._add_rnn_block(trial, block_index, features, activation)
            elif block_type == "cnn":
                structure = self._add_cnn_block(trial, block_index, features, activation)
            else:
                if block_index == 0:
                    features = vector_features
                structure = getattr(self, f"_add_{block_type}_block")(trial, block_index, features, activation)
            features = structure["out_features"]
            self.block_specs.append(structure)

        self.classifier_type = trial.suggest_categorical("classifier", ["softmax", "svm"])
        self.classifier = nn.Linear(features, num_classes)

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            xb = block(xb)
        return self.classifier(xb)

    def _add_mlp_block(
        self,
        trial: optuna.trial.Trial,
        block_index: int,
        features: int,
        activation: type[nn.Module],
    ) -> dict:
        structure = trial.suggest_categorical(f"block_{block_index}_mlp_structure", ["dense", "residual", "bottleneck"])
        width = trial.suggest_categorical(f"block_{block_index}_mlp_width", [32, 64, 128, 256])

        if structure == "dense":
            self.blocks.append(FlattenedBlock(nn.Sequential(nn.Linear(features, width), activation())))
            out_features = width
        elif structure == "residual":
            self.blocks.append(FlattenedBlock(ResidualMLPBlock(features, width, activation)))
            out_features = features
        else:
            bottleneck = max(8, width // 2)
            self.blocks.append(FlattenedBlock(nn.Sequential(
                nn.Linear(features, bottleneck),
                activation(),
                nn.Linear(bottleneck, width),
                activation(),
            )))
            out_features = width

        return {"type": "mlp", "structure": structure, "out_features": out_features}

    def _add_cnn_block(
        self,
        trial: optuna.trial.Trial,
        block_index: int,
        features: int,
        activation: type[nn.Module],
    ) -> dict:
        structure = trial.suggest_categorical(f"block_{block_index}_cnn_structure", ["single", "double", "wide"])
        channels = trial.suggest_categorical(f"block_{block_index}_cnn_channels", [4, 8, 16, 24, 32])

        if block_index == 0:
            in_channels = self.input_channels
            length = self.input_length
        else:
            in_channels = 1
            length = features

        if structure == "single":
            layers = [nn.Conv1d(in_channels, channels, kernel_size=3), activation()]
            length -= 2
        elif structure == "double":
            layers = [
                nn.Conv1d(in_channels, channels, kernel_size=3),
                activation(),
                nn.Conv1d(channels, channels, kernel_size=3),
                activation(),
            ]
            length -= 4
        else:
            layers = [nn.Conv1d(in_channels, channels, kernel_size=5, padding=2), activation()]

        self.blocks.append(CNNBlock(nn.Sequential(*layers)))
        return {
            "type": "cnn",
            "structure": structure,
            "channels": channels,
            "out_features": channels * length,
        }

    def _add_rnn_block(
        self,
        trial: optuna.trial.Trial,
        block_index: int,
        features: int,
        activation: type[nn.Module],
    ) -> dict:
        structure = trial.suggest_categorical(f"block_{block_index}_rnn_structure", ["rnn", "gru", "lstm"])
        hidden_size = trial.suggest_categorical(f"block_{block_index}_rnn_hidden", [16, 32, 64, 128])
        num_layers = trial.suggest_categorical(f"block_{block_index}_rnn_layers", [1, 2])
        rnn_cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[structure]

        self.blocks.append(RNNBlock(rnn_cls(features, hidden_size, num_layers=num_layers, batch_first=True), activation()))
        return {
            "type": "rnn",
            "structure": structure,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "out_features": hidden_size,
        }

    def _add_dbm_block(
        self,
        trial: optuna.trial.Trial,
        block_index: int,
        features: int,
        activation: type[nn.Module],
    ) -> dict:
        structure = trial.suggest_categorical(f"block_{block_index}_dbm_structure", ["two_layer", "three_layer", "funnel"])
        hidden = trial.suggest_categorical(f"block_{block_index}_dbm_hidden", [32, 64, 128, 256])

        if structure == "two_layer":
            layer_sizes = [features, hidden, hidden]
            out_features = hidden
        elif structure == "three_layer":
            layer_sizes = [features, hidden, hidden, hidden]
            out_features = hidden
        else:
            out_features = max(16, hidden // 2)
            layer_sizes = [features, hidden, out_features]

        self.blocks.append(FlattenedBlock(DeepBoltzmannBlock(layer_sizes)))
        return {"type": "dbm", "structure": structure, "out_features": out_features}

    def spec(self) -> dict:
        return {
            "activation": self.activation_name,
            "blocks": self.block_specs,
            "classifier": self.classifier_type,
            "model": repr(self),
        }


class FlattenedBlock(nn.Module):
    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        if xb.ndim > 2:
            xb = xb.flatten(1)
        return self.block(xb)


class RNNBlock(nn.Module):
    def __init__(self, rnn: nn.Module, activation: nn.Module) -> None:
        super().__init__()
        self.rnn = rnn
        self.activation = activation

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        if xb.ndim == 3:
            xb = xb.transpose(1, 2)
        else:
            xb = xb.unsqueeze(1)
        _, hidden = self.rnn(xb)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        return self.activation(hidden[-1])


def accuracy(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += len(yb)
    return correct / total


def loss_for(model: DynamicMultiNet, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if model.classifier_type == "svm":
        true_scores = logits.gather(1, labels.unsqueeze(1))
        margins = torch.clamp(logits - true_scores + 1.0, min=0.0)
        margins.scatter_(1, labels.unsqueeze(1), 0.0)
        return margins.sum(dim=1).mean()
    return nn.functional.cross_entropy(logits, labels)


def objective(trial: optuna.trial.Trial) -> float:
    net = DynamicMultiNet(trial, x.shape[1], x.shape[2], num_classes)
    model = net.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    epoch_bar = trange(EPOCHS, desc=f"Trial {trial.number}", leave=False)
    for _ in epoch_bar:
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_for(model, model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_bar.set_postfix(loss=f"{loss.item():.4f}")

    train_acc = accuracy(model, train_loader)
    val_acc = accuracy(model, val_loader)
    with TRIALS_PATH.open("a") as f:
        f.write(json.dumps({
            "trial": trial.number,
            "accuracies": {"train": train_acc, "validation": val_acc},
            "params": trial.params,
            "training": {"optimizer": "adam", "lr": LR, "batch_size": BATCH_SIZE, "epochs": EPOCHS},
            "architecture": net.spec(),
        }) + "\n")
    return val_acc
