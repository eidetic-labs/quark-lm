"""PyTorch attention helpers matching scalar transformer math."""

from __future__ import annotations

import math
from typing import Any


def torch_apply_rotary(
    rows: Any,
    config: dict[str, Any],
    torch: Any,
    positions: list[int] | None = None,
) -> Any:
    return torch.stack(
        [
            _apply_rotary_row(
                row, i if positions is None else positions[i], config, torch
            )
            for i, row in enumerate(rows)
        ]
    )


def torch_causal_attention(
    q: Any,
    k: Any,
    v: Any,
    config: dict[str, Any],
    torch: Any,
    position: int | None = None,
    runtime: dict[str, Any] | None = None,
) -> Any:
    if position is None:
        position = config["context_size"] - 1
    head_dim = config["embedding_dim"] // config["attention_heads"]
    # Opt-in (runtime['use_sdpa']) device-agnostic fused attention. Default off ->
    # the hand-rolled head, so the existing parity + the fake-torch double (which
    # lacks F.scaled_dot_product_attention) are untouched. SDPA computes the same
    # softmax((q.k)*scale).v and is validated within the parity band on real torch.
    head_fn = _attention_head_sdpa if (runtime and runtime.get("use_sdpa")) else _attention_head
    attended = []
    for head in range(config["attention_heads"]):
        attended.extend(head_fn(q, k, v, position, head, head_dim, torch))
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
            # position < 0 is a left-pad sentinel: rotate by the IDENTITY using
            # hard-coded host-float constants (NOT trig) so the pad row passes
            # through bit-exactly on f32/MPS. cos/sin stay host float64 scalars
            # multiplied into the (f32/MPS) row -> device-safe.
            if position < 0:
                cos_value, sin_value = 1.0, 0.0
            else:
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


def _attention_head_sdpa(
    q: Any,
    k: Any,
    v: Any,
    position: int,
    head: int,
    head_dim: int,
    torch: Any,
) -> list[Any]:
    """Single-query attention via F.scaled_dot_product_attention (device-agnostic).

    Same math as ``_attention_head`` -- softmax((q.k)*scale) over the past 0..position,
    weighted-sum of values -- but routed through the fused kernel (FlashAttention on
    CUDA, accelerated paths on MPS). Explicit scale = 1/sqrt(head_dim) matches the
    hand-rolled head exactly, so float64 output agrees within the parity band.
    """

    start = head * head_dim
    end = start + head_dim
    query = torch.stack([q[position][start:end]])
    keys = torch.stack([k[past][start:end] for past in range(position + 1)])
    values = torch.stack([v[past][start:end] for past in range(position + 1)])
    attended = torch.nn.functional.scaled_dot_product_attention(
        query, keys, values, scale=1.0 / math.sqrt(head_dim)
    )
    return [attended[0][dim] for dim in range(head_dim)]
