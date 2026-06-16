from torch import nn


FP32_BYTES_PER_PARAMETER = 4


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def fp32_size_kb(model: nn.Module) -> float:
    return parameter_count(model) * FP32_BYTES_PER_PARAMETER / 1024


def compactness_score(validation_accuracy: float, model: nn.Module, penalty_per_100k_params: float) -> float:
    return validation_accuracy - penalty_per_100k_params * parameter_count(model) / 100_000


def optimization_report(validation_accuracy: float, model: nn.Module, penalty_per_100k_params: float) -> dict:
    return {
        "score": compactness_score(validation_accuracy, model, penalty_per_100k_params),
        "parameter_count": parameter_count(model),
        "trainable_parameter_count": trainable_parameter_count(model),
        "size_kb_fp32": fp32_size_kb(model),
        "penalty_per_100k_params": penalty_per_100k_params,
    }
