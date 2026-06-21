"""PyTorch attention helpers matching scalar transformer math."""

from __future__ import annotations

import math
from typing import Any


def torch_apply_rotary(rows: Any, config: dict[str, Any], torch: Any) -> Any:
    return torch.stack(
        [
            _apply_rotary_row(row, position, config, torch)
            for position, row in enumerate(rows)
        ]
    )


def torch_causal_attention(
    q: Any,
    k: Any,
    v: Any,
    config: dict[str, Any],
    torch: Any,
    position: int | None = None,
) -> Any:
    if position is None:
        position = config["context_size"] - 1
    head_dim = config["embedding_dim"] // config["attention_heads"]
    attended = []
    for head in range(config["attention_heads"]):
        attended.extend(_attention_head(q, k, v, position, head, head_dim, torch))
    return torch.stack(attended)


def _apply_rotary_row(
    row: Any,
    position: int,
    config: dict[str, Any],
    torch: Any,
) -> Any:
    head_dim = config["embedding_dim"] // config["attention_heads"]
    output = [row[dim] for dim in range(config["embedding_dim"])]
    for head in range(config["attention_heads"]):
        start = head * head_dim
        for offset in range(0, head_dim - 1, 2):
            index = start + offset
            angle = position / (10000.0 ** (offset / max(head_dim, 1)))
            cos_value = math.cos(angle)
            sin_value = math.sin(angle)
            left = row[index]
            right = row[index + 1]
            output[index] = left * cos_value - right * sin_value
            output[index + 1] = left * sin_value + right * cos_value
    return torch.stack(output)


def _attention_head(
    q: Any,
    k: Any,
    v: Any,
    position: int,
    head: int,
    head_dim: int,
    torch: Any,
) -> list[Any]:
    start = head * head_dim
    end = start + head_dim
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.stack(
        [
            (q[position][start:end] * k[past][start:end]).sum() * scale
            for past in range(position + 1)
        ]
    )
    weights = torch.softmax(scores, dim=0)
    # Value aggregation as one (P)x(P, head_dim) matmul instead of head_dim*P scalar
    # nodes; weights @ values sums over past in the same order, so it is bit-exact at
    # float64 and within the validated parity band on device. 1D@2D keeps the
    # dependency-free parity double (which implements only vector@matrix) compatible.
    values = torch.stack([v[past][start:end] for past in range(position + 1)])
    attended = weights @ values
    return [attended[dim] for dim in range(head_dim)]
