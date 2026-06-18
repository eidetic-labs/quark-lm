"""PyTorch normalization helpers matching scalar transformer math."""

from __future__ import annotations

from typing import Any


def torch_layer_norm(
    values: Any,
    gain: Any,
    bias: Any,
    epsilon: float,
) -> Any:
    count = max(len(values), 1)
    mean = values.sum() / count
    centered = values - mean
    variance = (centered * centered).sum() / count
    scale = (variance + epsilon).pow(-0.5)
    return centered * scale * gain + bias


def torch_rms_norm(
    values: Any,
    gain: Any,
    epsilon: float,
) -> Any:
    count = max(len(values), 1)
    mean_square = (values * values).sum() / count
    scale = (mean_square + epsilon).pow(-0.5)
    return values * scale * gain
