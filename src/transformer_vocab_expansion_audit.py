"""Parity checks for append-only transformer vocabulary expansion."""

from __future__ import annotations

from typing import Any

from neural_char_ops import make_context


DEFAULT_CONTEXT_LIMIT = 8
DEFAULT_TOLERANCE = 1e-12


def audit_vocab_expansion_parity(
    *,
    base_model: Any,
    expanded_model: Any,
    base_tokenizer: Any,
    expanded_tokenizer: Any,
    train_text: str,
    context_size: int,
    context_limit: int = DEFAULT_CONTEXT_LIMIT,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    if not expanded_tokenizer.extends(base_tokenizer):
        return _failed("expanded_tokenizer_does_not_extend_base", [])
    contexts = _sample_contexts(
        base_tokenizer,
        train_text,
        context_size,
        context_limit,
    )
    failures = []
    old_size = base_tokenizer.vocab_size
    for index, context in enumerate(contexts):
        base_logits = base_model._forward_floats(context)[:old_size]
        expanded_logits = expanded_model._forward_floats(context)[:old_size]
        max_delta = max(
            (
                abs(base_logits[token_id] - expanded_logits[token_id])
                for token_id in range(old_size)
            ),
            default=0.0,
        )
        if max_delta > tolerance:
            failures.append(
                {
                    "context_index": index,
                    "max_old_logit_delta": max_delta,
                }
            )
    return {
        "kind": "vocab_expansion_parity_audit",
        "passed": not failures,
        "base_vocab_size": old_size,
        "expanded_vocab_size": expanded_tokenizer.vocab_size,
        "contexts_checked": len(contexts),
        "tolerance": tolerance,
        "failures": failures,
    }


def _sample_contexts(
    tokenizer: Any,
    train_text: str,
    context_size: int,
    limit: int,
) -> list[list[int]]:
    known_text = "".join(char for char in train_text if char in tokenizer.stoi)
    token_ids = tokenizer.encode(known_text)
    if not token_ids:
        return [[tokenizer.pad_id for _ in range(context_size)]]
    positions = _sample_positions(len(token_ids), limit)
    return [
        make_context(token_ids[:position], context_size, tokenizer.pad_id)
        for position in positions
    ]


def _sample_positions(token_count: int, limit: int) -> list[int]:
    if token_count <= limit:
        return list(range(1, token_count + 1))
    stride = max(token_count // limit, 1)
    positions = list(range(1, token_count + 1, stride))[:limit]
    if token_count not in positions:
        positions[-1] = token_count
    return positions


def _failed(reason: str, failures: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "kind": "vocab_expansion_parity_audit",
        "passed": False,
        "reason": reason,
        "failures": failures,
    }
