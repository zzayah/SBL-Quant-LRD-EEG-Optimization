"""Five fixed CNN baselines for motor-imagery EEG classification."""

from collections.abc import Callable

from torch import nn


class LogVariance(nn.Module):
    """Convert temporal feature maps into log-power features."""

    def forward(self, inputs):
        return inputs.square().mean(dim=-1).clamp_min(1e-7).log()


def _activation(name: str) -> nn.Module:
    if name == "elu":
        return nn.ELU()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unknown activation: {name}")


def _temporal_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    pool_size: int,
    dropout: float,
    activation: str,
    dilation: int = 1,
) -> list[nn.Module]:
    padding = dilation * (kernel_size - 1) // 2
    return [
        nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        ),
        nn.Conv1d(in_channels, out_channels, 1, bias=False),
        nn.BatchNorm1d(out_channels),
        _activation(activation),
        nn.AvgPool1d(pool_size),
        nn.Dropout(dropout),
    ]


def _build(
    input_channels: int,
    temporal_filters: int,
    temporal_kernel: int,
    spatial_multiplier: int,
    stem_pool: int,
    blocks: list[tuple[int, int, int, int]],
    dropout: float,
    activation: str,
    head: str = "flatten",
) -> nn.Sequential:
    spatial_filters = temporal_filters * spatial_multiplier
    layers: list[nn.Module] = [
        nn.Unflatten(1, (1, input_channels)),
        nn.Conv2d(
            1,
            temporal_filters,
            (1, temporal_kernel),
            padding="same",
            bias=False,
        ),
        nn.BatchNorm2d(temporal_filters),
        nn.Conv2d(
            temporal_filters,
            spatial_filters,
            (input_channels, 1),
            groups=temporal_filters,
            bias=False,
        ),
        nn.BatchNorm2d(spatial_filters),
        _activation(activation),
        nn.AvgPool2d((1, stem_pool)),
        nn.Dropout(dropout),
        nn.Flatten(1, 2),
    ]
    current_channels = spatial_filters
    feature_samples = 384 // stem_pool
    for out_channels, kernel_size, pool_size, dilation in blocks:
        layers.extend(
            _temporal_block(
                current_channels,
                out_channels,
                kernel_size,
                pool_size,
                dropout,
                activation,
                dilation,
            )
        )
        current_channels = out_channels
        feature_samples //= pool_size
    if head == "flatten":
        layers.extend(
            [nn.Flatten(), nn.Linear(current_channels * feature_samples, 2)]
        )
    elif head == "log_variance":
        layers.extend([LogVariance(), nn.Linear(current_channels, 2)])
    else:
        raise ValueError(f"Unknown head: {head}")
    return nn.Sequential(*layers)


def compact_eegnet(input_channels: int) -> nn.Sequential:
    return _build(input_channels, 8, 63, 2, 4, [(16, 15, 8, 1)], 0.25, "elu")


def wide_eegnet(input_channels: int) -> nn.Sequential:
    return _build(
        input_channels,
        16,
        31,
        2,
        4,
        [(32, 15, 2, 1), (48, 7, 2, 1)],
        0.3,
        "elu",
    )


def shallow_temporal(input_channels: int) -> nn.Sequential:
    return _build(
        input_channels,
        24,
        31,
        1,
        2,
        [(32, 31, 2, 1)],
        0.2,
        "relu",
        "log_variance",
    )


def deep_separable(input_channels: int) -> nn.Sequential:
    return _build(
        input_channels,
        8,
        31,
        2,
        2,
        [(16, 15, 2, 1), (24, 15, 2, 1), (32, 7, 2, 1), (48, 7, 2, 1)],
        0.3,
        "gelu",
    )


def dilated_temporal(input_channels: int) -> nn.Sequential:
    return _build(
        input_channels,
        16,
        31,
        2,
        2,
        [(32, 15, 2, 1), (48, 15, 2, 2), (64, 15, 2, 4)],
        0.3,
        "elu",
    )


def _scaled_eegnet(input_channels: int, width: int) -> nn.Sequential:
    return _build(
        input_channels,
        width,
        31,
        2,
        4,
        [(2 * width, 15, 2, 1), (3 * width, 7, 2, 1)],
        0.3,
        "elu",
    )


def scaled_xs(input_channels: int) -> nn.Sequential:
    return _scaled_eegnet(input_channels, 4)


def scaled_s(input_channels: int) -> nn.Sequential:
    return _scaled_eegnet(input_channels, 16)


def scaled_m(input_channels: int) -> nn.Sequential:
    return _scaled_eegnet(input_channels, 32)


def scaled_l(input_channels: int) -> nn.Sequential:
    return _scaled_eegnet(input_channels, 56)


def scaled_xl(input_channels: int) -> nn.Sequential:
    return _scaled_eegnet(input_channels, 80)


BASELINES: dict[str, Callable[[int], nn.Sequential]] = {
    "compact_eegnet": compact_eegnet,
    "wide_eegnet": wide_eegnet,
    "shallow_temporal": shallow_temporal,
    "deep_separable": deep_separable,
    "dilated_temporal": dilated_temporal,
}

SCALED_BASELINES: dict[str, Callable[[int], nn.Sequential]] = {
    "scaled_xs": scaled_xs,
    "scaled_s": scaled_s,
    "scaled_m": scaled_m,
    "scaled_l": scaled_l,
    "scaled_xl": scaled_xl,
}

BASELINE_SUITES = {"fixed": BASELINES, "scaled": SCALED_BASELINES}
