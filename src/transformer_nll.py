"""Transformer negative-log-likelihood evaluation helpers."""

from __future__ import annotations

from typing import Any

from neural_char_model import context_before


def average_nll(
    model: Any,
    ids: list[int],
    pad_id: int,
    limit: int | None = None,
) -> float:
    if not ids:
        return 0.0
    count = min(len(ids), limit) if limit else len(ids)
    total = 0.0
    for position in range(count):
        context = context_before(ids, position, model.config.context_size, pad_id)
        total += model.nll(context, ids[position])
    return total / count
