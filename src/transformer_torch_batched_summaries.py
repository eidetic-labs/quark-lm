"""Batched final-context summaries on the (B, D) last-position slice.

Mirror of ``transformer_torch_context_summary`` for the Tier-2 batched forward.
pad_id == 0 (tokenizer), so the non-pad mask is ``idx != 0`` -- the batched
reproduction of the scalar ``context[position] != 0`` gating. Position-collapsing
summaries (mean / projection over C, prompt-prefix mean, attention summary) reduce
safely with a pad mask; ``use_prompt_position_projection`` is fail-closed upstream
and never reaches this module.
"""

from __future__ import annotations

import math
from typing import Any

from transformer_torch_tensor_ops import torch_linear, torch_tensor


def add_batched_context_summaries(
    last: Any,
    x: Any,
    idx: Any,
    weights: dict[str, Any],
    config: dict[str, Any],
    torch: Any,
    runtime: dict[str, Any],
) -> Any:
    if config.get("use_context_mean"):
        last = last + _all_mean(x)
    if config.get("use_context_projection"):
        last = last + torch_linear(
            _all_mean(x),
            weights["context_projection_w"],
            weights["context_projection_b"],
            torch,
            runtime,
        )
    if config.get("use_prompt_prefix_projection"):
        last = _add_prompt_prefix_projection(last, x, idx, weights, config, torch, runtime)
    if config.get("use_prompt_attention_summary"):
        last = _add_prompt_attention_summary(last, x, weights, config, torch, runtime)
    return last


def _all_mean(x: Any) -> Any:
    """Mean over all C positions (unmasked), matching scalar ``_all_rows`` mean."""

    return x.mean(dim=1)


def _add_prompt_prefix_projection(last, x, idx, weights, config, torch, runtime):
    """Masked mean over the prefix x[:, :C-1] of non-pad rows, then one linear.

    Matches the scalar reference, which excludes the last position and divides by
    the per-example non-pad count. Examples with no prompt rows add nothing (the
    scalar returns ``hidden`` unchanged when ``prompt_rows`` is empty).
    """

    context_size = config["context_size"]
    prefix = x[:, : context_size - 1, :]
    mask = (idx[:, : context_size - 1] != 0).to(prefix.dtype).unsqueeze(-1)
    counts = mask.sum(dim=1)
    safe_counts = counts.clamp(min=1.0)
    summary = (prefix * mask).sum(dim=1) / safe_counts
    projected = torch_linear(
        summary,
        weights["prompt_prefix_projection_w"],
        weights["prompt_prefix_projection_b"],
        torch,
        runtime,
    )
    keep = (counts > 0).to(projected.dtype)
    return last + projected * keep


def _add_prompt_attention_summary(last, x, weights, config, torch, runtime):
    """Softmax attention over all C positions with a fixed query, then one linear."""

    query = torch_tensor(torch, weights["prompt_summary_query"], runtime)
    scale = 1.0 / math.sqrt(config["embedding_dim"])
    scores = (x * query).sum(dim=-1) * scale
    attention_weights = torch.softmax(scores, dim=1).unsqueeze(-1)
    summary = (x * attention_weights).sum(dim=1)
    return last + torch_linear(
        summary,
        weights["prompt_summary_w"],
        weights["prompt_summary_b"],
        torch,
        runtime,
    )
