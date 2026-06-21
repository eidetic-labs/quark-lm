"""PyTorch final-context summary helpers for transformer parity."""

from __future__ import annotations

import math
from typing import Any

from transformer_torch_tensor_ops import torch_linear, torch_tensor


def torch_add_final_context_summaries(
    hidden: Any,
    x: Any,
    context: list[int],
    weights: dict[str, Any],
    config: dict[str, Any],
    torch: Any,
    runtime: dict[str, Any],
) -> Any:
    if config.get("use_context_mean"):
        hidden = hidden + _mean_rows(_all_rows(x, config), config, torch)
    if config.get("use_context_projection"):
        hidden = hidden + torch_linear(
            _mean_rows(_all_rows(x, config), config, torch),
            weights["context_projection_w"],
            weights["context_projection_b"],
            torch,
            runtime,
        )
    if config.get("use_prompt_prefix_projection"):
        hidden = _add_prompt_prefix_projection(
            hidden, x, context, weights, config, torch, runtime
        )
    if config.get("use_prompt_position_projection"):
        hidden = _add_prompt_position_projection(
            hidden, x, context, weights, config, torch, runtime
        )
    if config.get("use_prompt_attention_summary"):
        hidden = _add_prompt_attention_summary(
            hidden, x, weights, config, torch, runtime
        )
    return hidden


def _add_prompt_prefix_projection(
    hidden: Any,
    x: Any,
    context: list[int],
    weights: dict[str, Any],
    config: dict[str, Any],
    torch: Any,
    runtime: dict[str, Any],
) -> Any:
    prompt_rows = _prompt_rows(x, context, config["context_size"] - 1)
    if not prompt_rows:
        return hidden
    return hidden + torch_linear(
        _mean_rows(prompt_rows, config, torch),
        weights["prompt_prefix_projection_w"],
        weights["prompt_prefix_projection_b"],
        torch,
        runtime,
    )


def _add_prompt_position_projection(
    hidden: Any,
    x: Any,
    context: list[int],
    weights: dict[str, Any],
    config: dict[str, Any],
    torch: Any,
    runtime: dict[str, Any],
) -> Any:
    prompt_positions = _prompt_positions(x, context, config["context_size"] - 1)
    if not prompt_positions:
        return hidden
    # Per-position matmul + accumulate. This replaces a triple Python loop that built
    # ~positions*embedding_dim*output_dim scalar-mul graph nodes per forward (the
    # profiled hot spot) with one (E)x(E,O) matmul per prompt position; same math, so
    # it stays within the validated parity band and rank-invariance contract. Uses the
    # `@` op (not einsum) so the dependency-free parity test double also supports it.
    weight = weights["prompt_position_projection_w"]
    contributions = [
        row @ torch_tensor(torch, weight[position], runtime)
        for position, row in prompt_positions
    ]
    total = contributions[0]
    for contribution in contributions[1:]:
        total = total + contribution
    projected = total / len(prompt_positions) + torch_tensor(
        torch, weights["prompt_position_projection_b"], runtime
    )
    return hidden + projected * config["prompt_position_projection_scale"]


def _add_prompt_attention_summary(
    hidden: Any,
    x: Any,
    weights: dict[str, Any],
    config: dict[str, Any],
    torch: Any,
    runtime: dict[str, Any],
) -> Any:
    rows = _all_rows(x, config)
    query = torch_tensor(torch, weights["prompt_summary_query"], runtime)
    scale = 1.0 / math.sqrt(config["embedding_dim"])
    scores = torch.stack([(query * row).sum() * scale for row in rows])
    attention_weights = torch.softmax(scores, dim=0)
    attention_summary = torch.stack(
        [
            torch.stack(
                [attention_weights[index] * row[dim] for index, row in enumerate(rows)]
            ).sum()
            for dim in range(config["embedding_dim"])
        ]
    )
    return hidden + torch_linear(
        attention_summary,
        weights["prompt_summary_w"],
        weights["prompt_summary_b"],
        torch,
        runtime,
    )


def _mean_rows(rows: list[Any], config: dict[str, Any], torch: Any) -> Any:
    return torch.stack(
        [
            torch.stack([row[dim] for row in rows]).sum() / len(rows)
            for dim in range(config["embedding_dim"])
        ]
    )


def _all_rows(x: Any, config: dict[str, Any]) -> list[Any]:
    return [x[position] for position in range(config["context_size"])]


def _prompt_rows(x: Any, context: list[int], last_position: int) -> list[Any]:
    return [
        x[position]
        for position in range(last_position)
        if context[position] != 0
    ]


def _prompt_positions(
    x: Any,
    context: list[int],
    last_position: int,
) -> list[tuple[int, Any]]:
    return [
        (position, x[position])
        for position in range(last_position)
        if context[position] != 0
    ]
