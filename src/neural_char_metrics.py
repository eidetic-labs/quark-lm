"""Evaluation metrics for the dependency-free character model."""

from __future__ import annotations

from typing import Any

from neural_char_ops import context_before, make_context
from tokenizer import CharTokenizer


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


def continuation_nll(
    model: Any,
    tokenizer: CharTokenizer,
    prompt: str,
    target: str,
) -> float:
    prompt_ids = tokenizer.encode(prompt)
    target_ids = tokenizer.encode(target)
    ids = prompt_ids[:]
    total = 0.0
    for target_id in target_ids:
        context = make_context(ids, model.config.context_size, tokenizer.pad_id)
        total += model.nll(context, target_id)
        ids.append(target_id)
    return total / max(len(target_ids), 1)
