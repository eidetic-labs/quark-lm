"""Supported-profile checks for minimal PyTorch transformer parity."""

from __future__ import annotations

from typing import Any


def minimal_forward_unsupported_reason(config: dict[str, Any]) -> str | None:
    unsupported_flags = [
        "use_gated_mlp",
        "tie_output_embeddings",
        "use_rotary_positions",
        "use_kv_cache_path",
        "use_context_mean",
        "use_context_projection",
        "use_prompt_prefix_projection",
        "use_prompt_position_projection",
        "use_prompt_attention_summary",
    ]
    if config.get("num_layers") != 1:
        return "minimal PyTorch parity supports exactly one transformer layer"
    if config.get("attention_heads") != 1:
        return "minimal PyTorch parity supports exactly one attention head"
    for flag in unsupported_flags:
        if config.get(flag):
            return f"minimal PyTorch parity does not support {flag}"
    return None
