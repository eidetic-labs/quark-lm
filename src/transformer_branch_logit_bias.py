"""Bias summaries for branch logit-prior diagnostics."""

from __future__ import annotations

from collections import Counter
from typing import Any

from tokenizer import CharTokenizer


def branch_logit_top_token_items(
    counter: Counter[int],
    tokenizer: CharTokenizer,
    limit: int = 12,
) -> list[dict[str, Any]]:
    return [
        {"value": tokenizer.itos[index], "count": count}
        for index, count in counter.most_common(limit)
    ]


def branch_logit_bias_rankings(
    model: Any,
    tokenizer: CharTokenizer,
) -> dict[int, int]:
    return {
        index: rank + 1
        for rank, index in enumerate(
            sorted(
                range(model.config.vocab_size),
                key=lambda token_id: (
                    -model.bout[token_id].data,
                    tokenizer.itos[token_id],
                    token_id,
                ),
            )
        )
    }


def missing_branch_target_ids(
    target_counts: Counter[int],
    predicted_counts: Counter[int],
    tokenizer: CharTokenizer,
) -> list[int]:
    target_token_ids = set(target_counts)
    predicted_token_ids = set(predicted_counts)
    return sorted(
        target_token_ids - predicted_token_ids,
        key=lambda token_id: (
            -target_counts[token_id],
            tokenizer.itos[token_id],
            token_id,
        ),
    )


def top_branch_bias_ids(model: Any, tokenizer: CharTokenizer, limit: int = 12) -> list[int]:
    return sorted(
        range(model.config.vocab_size),
        key=lambda token_id: (
            -model.bout[token_id].data,
            tokenizer.itos[token_id],
            token_id,
        ),
    )[:limit]


def branch_logit_bias_summary(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "avg": sum(values) / len(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }


def branch_logit_top_bias_tokens(
    model: Any,
    tokenizer: CharTokenizer,
    bias_rankings: dict[int, int],
    limit: int = 12,
) -> list[dict[str, Any]]:
    return [
        {
            "token": tokenizer.itos[token_id],
            "bias": model.bout[token_id].data,
            "rank": bias_rankings[token_id],
        }
        for token_id in top_branch_bias_ids(model, tokenizer, limit)
    ]
