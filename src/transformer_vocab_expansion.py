"""Compositional vocabulary expansion for transformer checkpoints."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def expand_weights_for_tokenizer(
    weights: dict[str, Any],
    base_tokenizer: Any,
    expanded_tokenizer: Any,
) -> dict[str, Any]:
    if not expanded_tokenizer.extends(base_tokenizer):
        raise ValueError("expanded tokenizer must preserve base token ids")
    expanded = deepcopy(weights)
    old_size = base_tokenizer.vocab_size
    new_size = expanded_tokenizer.vocab_size
    if new_size < old_size:
        raise ValueError("expanded tokenizer cannot shrink vocabulary")
    if new_size == old_size:
        return expanded

    token_embeddings = expanded["token_embeddings"]
    wout = expanded["wout"]
    bout = expanded["bout"]
    for token_id in range(old_size, new_size):
        part_ids = _new_token_part_ids(token_id, base_tokenizer, expanded_tokenizer)
        token_embeddings.append(_average_rows(token_embeddings, part_ids))
        for row in wout:
            row.append(_average_values(row, part_ids))
        bout.append(0.0)
    return expanded


def _new_token_part_ids(
    token_id: int,
    base_tokenizer: Any,
    expanded_tokenizer: Any,
) -> list[int]:
    token = expanded_tokenizer.itos[token_id]
    try:
        return base_tokenizer.encode(token)
    except ValueError:
        pass

    merge_parts = _merge_rule_part_ids(token_id, token, expanded_tokenizer)
    if merge_parts is not None:
        return merge_parts
    if len(token) == 1:
        return _atomic_seed_ids(base_tokenizer)
    char_ids = [
        expanded_tokenizer.stoi[char]
        for char in token
        if char in expanded_tokenizer.stoi
    ]
    if char_ids and all(part_id < token_id for part_id in char_ids):
        return char_ids
    raise ValueError(f"new token {token!r} cannot be decomposed safely")


def _merge_rule_part_ids(
    token_id: int,
    token: str,
    expanded_tokenizer: Any,
) -> list[int] | None:
    for rule in getattr(expanded_tokenizer, "merge_rules", []):
        if rule.token != token:
            continue
        part_ids = [expanded_tokenizer.stoi[rule.left], expanded_tokenizer.stoi[rule.right]]
        if all(part_id < token_id for part_id in part_ids):
            return part_ids
    return None


def _atomic_seed_ids(base_tokenizer: Any) -> list[int]:
    non_pad_ids = [
        token_id
        for token_id in range(base_tokenizer.vocab_size)
        if token_id != base_tokenizer.pad_id
    ]
    return non_pad_ids or [base_tokenizer.pad_id]


def _average_rows(rows: list[list[float]], indexes: list[int]) -> list[float]:
    if not indexes:
        raise ValueError("new token must decompose into existing tokens")
    width = len(rows[0])
    return [
        sum(rows[index][dim] for index in indexes) / len(indexes)
        for dim in range(width)
    ]


def _average_values(row: list[float], indexes: list[int]) -> float:
    if not indexes:
        raise ValueError("new token must decompose into existing tokens")
    return sum(row[index] for index in indexes) / len(indexes)
