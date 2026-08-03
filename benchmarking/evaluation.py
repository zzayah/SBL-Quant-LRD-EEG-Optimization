"""Detailed validation metrics for fixed EEG baselines."""

import io
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Return aggregate, per-class, per-subject, and efficiency metrics."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    targets_all = []
    predictions_all = []
    loss_sum = 0.0

    with torch.no_grad():
        for signals, targets in loader:
            signals = signals.to(device)
            targets = targets.to(device)
            logits = model(signals)
            loss_sum += float(criterion(logits, targets))
            targets_all.append(targets.cpu().numpy())
            predictions_all.append(logits.argmax(1).cpu().numpy())

    targets = np.concatenate(targets_all)
    predictions = np.concatenate(predictions_all)
    confusion = np.zeros((2, 2), dtype=np.int64)
    for target, prediction in zip(targets, predictions, strict=True):
        confusion[target, prediction] += 1

    recalls = np.diag(confusion) / confusion.sum(axis=1)
    precisions = np.diag(confusion) / np.maximum(confusion.sum(axis=0), 1)
    f1 = 2 * precisions * recalls / np.maximum(precisions + recalls, 1e-12)
    subjects = loader.dataset.trial_subjects
    per_subject = {
        str(subject): float(np.mean(predictions[subjects == subject] == targets[subjects == subject]))
        for subject in np.unique(subjects)
    }

    sample_signals, _ = next(iter(loader))
    sample_signals = sample_signals.to(device)
    with torch.no_grad():
        for _ in range(5):
            model(sample_signals)
        _synchronize(device)
        start = time.perf_counter()
        for _ in range(30):
            model(sample_signals)
        _synchronize(device)
    elapsed = time.perf_counter() - start
    trials_measured = 30 * len(sample_signals)

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return {
        "samples": int(len(targets)),
        "loss": loss_sum / len(targets),
        "accuracy": float(np.mean(predictions == targets)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1)),
        "left_recall": float(recalls[0]),
        "right_recall": float(recalls[1]),
        "left_precision": float(precisions[0]),
        "right_precision": float(precisions[1]),
        "confusion_matrix": confusion.tolist(),
        "per_subject_accuracy": per_subject,
        "inference_ms_per_trial": 1000 * elapsed / trials_measured,
        "inference_trials_per_second": trials_measured / elapsed,
        "serialized_state_bytes": buffer.getbuffer().nbytes,
    }
