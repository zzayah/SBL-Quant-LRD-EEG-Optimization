import json
from pathlib import Path

import numpy as np


DATA_PATH = Path("data/driver-drowsiness/orosco/dataset/orosco_windowed_balanced.npz")
TRAIN_FRACTION = 0.8
MAX_FEATURES = None
MAX_ITER = 50
TOLERANCE = 1e-4
ACTIVE_ALPHA_THRESHOLD = 1e4


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    x = data["X"].astype(np.float64).reshape(len(data["X"]), -1)
    y = data["Y"].astype(np.int64)
    return x, y


def split_data(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_train = int(TRAIN_FRACTION * len(x))
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:]


def select_features(x_train: np.ndarray, x_val: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if MAX_FEATURES is None or MAX_FEATURES >= x_train.shape[1]:
        selected = np.arange(x_train.shape[1])
    else:
        selected = np.argsort(x_train.var(axis=0))[-MAX_FEATURES:]
    return x_train[:, selected], x_val[:, selected], selected


def standardize(x_train: np.ndarray, x_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    return (x_train - mean) / std, (x_val - mean) / std


def add_bias(x: np.ndarray) -> np.ndarray:
    return np.column_stack([x, np.ones(len(x))])


def encode_labels(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(f"SBL classifier expects exactly 2 classes, found {len(classes)}")
    targets = np.where(y == classes[1], 1.0, -1.0)
    return targets, classes


def fit_sbl_classifier(x_train: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, int, bool]:
    design = add_bias(x_train)
    n_samples, n_features = design.shape

    alpha = np.ones(n_features)
    alpha[-1] = 1e-6
    beta = 1.0
    weights = np.zeros(n_features)
    converged = False

    for iteration in range(1, MAX_ITER + 1):
        previous_weights = weights.copy()
        precision = np.diag(alpha) + beta * design.T @ design
        covariance = np.linalg.inv(precision)
        weights = beta * covariance @ design.T @ targets

        gamma = 1.0 - alpha * np.diag(covariance)
        updated_alpha = gamma / np.maximum(weights**2, 1e-12)
        updated_alpha = np.clip(updated_alpha, 1e-8, 1e12)
        updated_alpha[-1] = 1e-6
        alpha = updated_alpha

        residual = targets - design @ weights
        beta_numerator = max(n_samples - gamma[:-1].sum(), 1e-8)
        beta = beta_numerator / max(float(residual @ residual), 1e-8)
        beta = float(np.clip(beta, 1e-8, 1e8))

        weight_change = np.linalg.norm(weights - previous_weights) / max(np.linalg.norm(previous_weights), 1.0)
        if weight_change < TOLERANCE:
            converged = True
            break

    return weights, alpha, beta, iteration, converged


def predict(x: np.ndarray, weights: np.ndarray, classes: np.ndarray) -> np.ndarray:
    scores = add_bias(x) @ weights
    return np.where(scores >= 0, classes[1], classes[0])


def main() -> None:
    x, y = load_data(DATA_PATH)
    x_train, y_train, x_val, y_val = split_data(x, y)
    x_train, x_val, selected_features = select_features(x_train, x_val)
    x_train, x_val = standardize(x_train, x_val)

    targets, classes = encode_labels(y_train)
    weights, alpha, beta, iterations, converged = fit_sbl_classifier(x_train, targets)
    train_accuracy = float((predict(x_train, weights, classes) == y_train).mean())
    validation_accuracy = float((predict(x_val, weights, classes) == y_val).mean())
    active_features = int((alpha[:-1] < ACTIVE_ALPHA_THRESHOLD).sum())

    print(json.dumps({
        "data": str(DATA_PATH),
        "split": {
            "train_fraction": TRAIN_FRACTION,
            "train_samples": len(x_train),
            "validation_samples": len(x_val),
        },
        "features": {
            "input_features": x.shape[1],
            "selected_features": int(len(selected_features)),
            "active_features": active_features,
            "active_alpha_threshold": ACTIVE_ALPHA_THRESHOLD,
        },
        "training": {
            "method": "ard_sparse_bayesian_linear_classifier",
            "max_iter": MAX_ITER,
            "iterations": iterations,
            "converged": converged,
            "beta": beta,
            "classes": classes.tolist(),
        },
        "accuracies": {
            "train": train_accuracy,
            "validation": validation_accuracy,
        },
    }, indent=2))


if __name__ == "__main__":
    main()
