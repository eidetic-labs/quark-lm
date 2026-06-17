"""Transformer probability and loss helpers."""

from __future__ import annotations

import math

from autograd import Scalar


def softmax_scalars(logits: list[Scalar]) -> list[Scalar]:
    max_logit = max(logit.data for logit in logits)
    exps = [(logit - max_logit).exp() for logit in logits]
    total = Scalar(0.0)
    for value in exps:
        total = total + value
    return [value / total for value in exps]


def softmax_floats(logits: list[float]) -> list[float]:
    max_logit = max(logits)
    exps = [math.exp(value - max_logit) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


def cross_entropy_scalars(logits: list[Scalar], target: int) -> Scalar:
    probs = softmax_scalars(logits)
    return -probs[target].log()
