"""Stateless operations for the dependency-free character model."""

from __future__ import annotations

import math
import random


def softmax(logits: list[float]) -> list[float]:
    max_logit = max(logits)
    exps = [math.exp(value - max_logit) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


def sample_from_probs(probs: list[float], temperature: float, rng: random.Random) -> int:
    adjusted = [pow(max(prob, 1e-12), 1.0 / temperature) for prob in probs]
    total = sum(adjusted)
    threshold = rng.random() * total
    running = 0.0
    for index, prob in enumerate(adjusted):
        running += prob
        if running >= threshold:
            return index
    return len(probs) - 1


def make_context(ids: list[int], context_size: int, pad_id: int) -> list[int]:
    if len(ids) >= context_size:
        return ids[-context_size:]
    return [pad_id] * (context_size - len(ids)) + ids


def context_before(
    ids: list[int],
    position: int,
    context_size: int,
    pad_id: int,
) -> list[int]:
    start = max(0, position - context_size)
    return make_context(ids[start:position], context_size, pad_id)
