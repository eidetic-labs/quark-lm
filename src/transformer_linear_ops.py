"""Transformer linear algebra helpers."""

from __future__ import annotations

from autograd import Scalar


def linear_scalars(
    inputs: list[Scalar],
    weights: list[list[Scalar]],
    bias: list[Scalar],
) -> list[Scalar]:
    outputs: list[Scalar] = []
    for output_index, bias_value in enumerate(bias):
        total = bias_value
        for input_index, value in enumerate(inputs):
            total = total + value * weights[input_index][output_index]
        outputs.append(total)
    return outputs


def linear_floats(
    inputs: list[float],
    weights: list[list[float]],
    bias: list[float],
) -> list[float]:
    outputs: list[float] = []
    for output_index, bias_value in enumerate(bias):
        total = bias_value
        for input_index, value in enumerate(inputs):
            total += value * weights[input_index][output_index]
        outputs.append(total)
    return outputs


def dot_scalars(left: list[Scalar], right: list[Scalar]) -> Scalar:
    total = Scalar(0.0)
    for left_value, right_value in zip(left, right):
        total = total + left_value * right_value
    return total


def dot_floats(left: list[float], right: list[float]) -> float:
    return sum(left_value * right_value for left_value, right_value in zip(left, right))
