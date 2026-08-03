"""Classification metrics for motor-imagery EEG models."""

import torch
from torch import nn
from torch.utils.data import DataLoader


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> dict[str, float | int]:
    """Evaluate loss, accuracy, and balanced accuracy."""
    model.eval()
    criterion = criterion or nn.CrossEntropyLoss()
    loss_sum = 0.0
    correct = 0
    class_correct = torch.zeros(2, dtype=torch.long)
    class_total = torch.zeros(2, dtype=torch.long)

    with torch.no_grad():
        for signals, targets in loader:
            signals = signals.to(device)
            targets = targets.to(device)
            logits = model(signals)
            loss_sum += float(criterion(logits, targets)) * len(targets)
            predictions = logits.argmax(dim=1)
            correct += int((predictions == targets).sum())
            for class_id in range(2):
                mask = targets == class_id
                class_total[class_id] += int(mask.sum())
                class_correct[class_id] += int((predictions[mask] == class_id).sum())

    samples = int(class_total.sum())
    if samples == 0 or torch.any(class_total == 0):
        raise ValueError("Evaluation requires samples from both classes")
    recall = class_correct.float() / class_total.float()
    return {
        "samples": samples,
        "loss": loss_sum / samples,
        "accuracy": correct / samples,
        "balanced_accuracy": float(recall.mean()),
        "left_recall": float(recall[0]),
        "right_recall": float(recall[1]),
    }
