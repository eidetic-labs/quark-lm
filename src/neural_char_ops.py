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


# Sentinel absolute position for a left-pad slot. A future absolute-RoPE consumer
# maps it to the identity rotation; it must never be read as a real sequence index.
POSITION_PAD_SENTINEL = -1


def make_context(ids: list[int], context_size: int, pad_id: int) -> list[int]:
    if len(ids) >= context_size:
        return ids[-context_size:]
    return [pad_id] * (context_size - len(ids)) + ids


def make_context_positioned(
    ids: list[int], context_size: int, pad_id: int
) -> tuple[list[int], list[int]]:
    """Right-anchored window plus each slot's ABSOLUTE sequence index.

    A pure superset of ``make_context``: the returned window content is byte-identical
    (the existing right-anchored geometry is unchanged), and ``positions[i]`` is the
    true 0-based index of ``context[i]`` in the full id stream. Left-pad slots carry
    ``POSITION_PAD_SENTINEL``. Because real positions are absolute -- they grow with
    the stream and never re-index as the window slides -- the key/value at a position
    is write-once, which is the precondition an append-valid KV cache needs (Phase 1+).
    """

    context = make_context(ids, context_size, pad_id)
    if len(ids) >= context_size:
        start = len(ids) - context_size
        return context, list(range(start, start + context_size))
    pad = context_size - len(ids)
    return context, [POSITION_PAD_SENTINEL] * pad + list(range(len(ids)))


def context_before(
    ids: list[int],
    position: int,
    context_size: int,
    pad_id: int,
) -> list[int]:
    start = max(0, position - context_size)
    return make_context(ids[start:position], context_size, pad_id)
