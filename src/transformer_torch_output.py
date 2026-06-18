"""PyTorch output-head helpers for transformer parity."""

from __future__ import annotations

from typing import Any

from transformer_torch_tensor_ops import torch_linear


def torch_output_logits(
    hidden: Any,
    weights: dict[str, Any],
    config: dict[str, Any],
    torch: Any,
    runtime: dict[str, Any],
) -> Any:
    output_weights = weights["wout"]
    if config.get("tie_output_embeddings"):
        output_weights = _transpose_token_embeddings(weights["token_embeddings"])
    return torch_linear(hidden, output_weights, weights["bout"], torch, runtime)


def _transpose_token_embeddings(
    token_embeddings: list[list[float]],
) -> list[list[float]]:
    return [
        [token_embeddings[token_id][dim] for token_id in range(len(token_embeddings))]
        for dim in range(len(token_embeddings[0]))
    ]
