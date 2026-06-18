"""Long-answer diagnostics for transformer generation."""

from __future__ import annotations

import math
import time
from typing import Any

from neural_char_ops import make_context
from transformer_math import softmax_floats
from transformer_model import GenerationConfig


def answer_diagnostics(
    model: Any,
    tokenizer: Any,
    prompt: str,
    target: str,
    generation_config: GenerationConfig | None = None,
) -> dict[str, Any]:
    generation_config = generation_config or GenerationConfig()
    target_ids = tokenizer.encode(target)
    started = time.perf_counter()
    generation = model.generate_with_trace(
        tokenizer,
        prompt,
        len(target_ids),
        generation_config,
    )
    generation_time_ms = (time.perf_counter() - started) * 1000.0
    completion = generation["text"]
    completion_ids = tokenizer.encode(completion)
    return {
        "prompt": prompt,
        "target": target,
        "completion": completion,
        "exact_match": completion == target,
        "target_token_count": len(target_ids),
        "completion_token_count": len(completion_ids),
        "first_drift_index": first_drift_index(target, completion),
        "per_token_nll": per_token_nll(model, tokenizer, prompt, target),
        "generation_time_ms": generation_time_ms,
        "generation_trace_steps": len(generation["trace"]),
        "generation_config": generation["generation_config"],
    }


def per_token_nll(
    model: Any,
    tokenizer: Any,
    prompt: str,
    target: str,
) -> list[dict[str, Any]]:
    ids = tokenizer.encode(prompt)
    records: list[dict[str, Any]] = []
    for offset, token_id in enumerate(tokenizer.encode(target)):
        context = make_context(ids, model.config.context_size, tokenizer.pad_id)
        probs = softmax_floats(model._forward_floats(context))
        nll = -math.log(max(probs[token_id], 1e-12))
        records.append(
            {
                "position": offset,
                "token_id": token_id,
                "token": tokenizer.itos[token_id],
                "nll": nll,
            }
        )
        ids.append(token_id)
    return records


def first_drift_index(expected: str, actual: str) -> int | None:
    for index, (left, right) in enumerate(zip(expected, actual)):
        if left != right:
            return index
    if len(expected) == len(actual):
        return None
    return min(len(expected), len(actual))
