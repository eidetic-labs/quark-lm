"""Transformer normalization helpers."""

from __future__ import annotations

import math

from autograd import Scalar


def layer_norm_scalars(
    values: list[Scalar],
    gain: list[Scalar],
    bias: list[Scalar],
    epsilon: float,
) -> list[Scalar]:
    count = max(len(values), 1)
    mean = Scalar(0.0)
    for value in values:
        mean = mean + value
    mean = mean / count
    variance = Scalar(0.0)
    for value in values:
        centered = value - mean
        variance = variance + centered * centered
    variance = variance / count
    scale = (variance + epsilon).pow(-0.5)
    return [
        (value - mean) * scale * gain[index] + bias[index]
        for index, value in enumerate(values)
    ]


def layer_norm_floats(
    values: list[float],
    gain: list[float],
    bias: list[float],
    epsilon: float,
) -> list[float]:
    count = max(len(values), 1)
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / count
    scale = 1.0 / math.sqrt(variance + epsilon)
    return [
        (value - mean) * scale * gain[index] + bias[index]
        for index, value in enumerate(values)
    ]


def rms_norm_scalars(
    values: list[Scalar],
    gain: list[Scalar],
    epsilon: float,
) -> list[Scalar]:
    count = max(len(values), 1)
    mean_square = Scalar(0.0)
    for value in values:
        mean_square = mean_square + value * value
    mean_square = mean_square / count
    scale = (mean_square + epsilon).pow(-0.5)
    return [
        value * scale * gain[index]
        for index, value in enumerate(values)
    ]


def rms_norm_floats(
    values: list[float],
    gain: list[float],
    epsilon: float,
) -> list[float]:
    count = max(len(values), 1)
    mean_square = sum(value * value for value in values) / count
    scale = 1.0 / math.sqrt(mean_square + epsilon)
    return [
        value * scale * gain[index]
        for index, value in enumerate(values)
    ]
