"""Minimal PyTorch transformer block math for parity fixtures."""

from __future__ import annotations

from typing import Any

from transformer_torch_attention import torch_apply_rotary, torch_causal_attention
from transformer_torch_context_summary import torch_add_final_context_summaries
from transformer_torch_feedforward import torch_feed_forward
from transformer_torch_norms import torch_layer_norm, torch_rms_norm
from transformer_torch_output import torch_output_logits
from transformer_torch_tensor_ops import torch_linear, torch_tensor


def torch_minimal_logits(
    context: list[int],
    fixture: dict[str, Any],
    torch: Any,
    runtime: dict[str, Any],
) -> Any:
    weights = fixture["weights"]
    config = fixture["model_config"]
    token_embeddings = torch_tensor(torch, weights["token_embeddings"], runtime)
    position_embeddings = torch_tensor(torch, weights["position_embeddings"], runtime)
    x = torch.stack(
        [
            token_embeddings[token_id] + position_embeddings[position]
            for position, token_id in enumerate(context)
        ]
    )
    blocks = _transformer_blocks(weights)
    for block in blocks[:-1]:
        x = _forward_full_block(x, block, config, torch, runtime)
    final_hidden = _forward_final_block(
        x,
        context,
        blocks[-1],
        weights,
        config,
        torch,
        runtime,
    )
    final_hidden = _finalize_hidden(final_hidden, weights, config, torch, runtime)
    return torch_output_logits(final_hidden, weights, config, torch, runtime)


def _transformer_blocks(weights: dict[str, Any]) -> list[dict[str, Any]]:
    return [weights] + list(weights.get("extra_layers", []))


def _forward_full_block(
    x: Any,
    block: dict[str, Any],
    config: dict[str, Any],
    torch: Any,
    runtime: dict[str, Any],
) -> Any:
    q, k, v = _attention_projections(x, block, config, torch, runtime)
    return torch.stack(
        [
            torch_feed_forward(
                _attention_hidden_at_position(
                    x,
                    block,
                    q,
                    k,
                    v,
                    position,
                    config,
                    torch,
                    runtime,
                ),
                block,
                config,
                torch,
                runtime,
            )
            for position in range(config["context_size"])
        ]
    )


def _forward_final_block(
    x: Any,
    context: list[int],
    block: dict[str, Any],
    weights: dict[str, Any],
    config: dict[str, Any],
    torch: Any,
    runtime: dict[str, Any],
) -> Any:
    q, k, v = _attention_projections(x, block, config, torch, runtime)
    hidden = _attention_hidden_at_position(
        x,
        block,
        q,
        k,
        v,
        config["context_size"] - 1,
        config,
        torch,
        runtime,
    )
    hidden = torch_add_final_context_summaries(
        hidden,
        x,
        context,
        weights,
        config,
        torch,
        runtime,
    )
    return torch_feed_forward(hidden, block, config, torch, runtime)


def _attention_projections(
    x: Any,
    block: dict[str, Any],
    config: dict[str, Any],
    torch: Any,
    runtime: dict[str, Any],
) -> tuple[Any, Any, Any]:
    attention_input = _attention_input(x, block, config, torch, runtime)
    q = _project_rows(attention_input, block, "wq", "bq", torch, runtime)
    k = _project_rows(attention_input, block, "wk", "bk", torch, runtime)
    v = _project_rows(attention_input, block, "wv", "bv", torch, runtime)
    if config.get("use_rotary_positions"):
        # Consumption gated on use_absolute_rope: absent the flag, positions are
        # dropped -> slot-keyed (enumerate), byte-identical to the pre-absolute path.
        positions = runtime.get("abs_positions") if config.get("use_absolute_rope") else None
        q = torch_apply_rotary(q, config, torch, positions)
        k = torch_apply_rotary(k, config, torch, positions)
    return q, k, v


def _attention_hidden_at_position(
    x: Any,
    block: dict[str, Any],
    q: Any,
    k: Any,
    v: Any,
    position: int,
    config: dict[str, Any],
    torch: Any,
    runtime: dict[str, Any],
) -> Any:
    attended = torch_causal_attention(q, k, v, config, torch, position, runtime)
    projected = torch_linear(attended, block["wo"], block["bo"], torch, runtime)
    return x[position] + projected


def _project_rows(
    rows: Any,
    weights: dict[str, Any],
    weight_key: str,
    bias_key: str,
    torch: Any,
    runtime: dict[str, Any],
) -> Any:
    return torch.stack(
        [
            torch_linear(row, weights[weight_key], weights[bias_key], torch, runtime)
            for row in rows
        ]
    )


def _attention_input(
    x: Any,
    weights: dict[str, Any],
    config: dict[str, Any],
    torch: Any,
    runtime: dict[str, Any],
) -> Any:
    if not config.get("use_pre_layer_norm"):
        return x
    gain = torch_tensor(torch, weights["ln1_gain"], runtime)
    if config.get("use_rms_norm"):
        return torch.stack(
            [torch_rms_norm(row, gain, config["layer_norm_epsilon"]) for row in x]
        )
    bias = torch_tensor(torch, weights["ln1_bias"], runtime)
    return torch.stack(
        [
            torch_layer_norm(row, gain, bias, config["layer_norm_epsilon"])
            for row in x
        ]
    )


def _finalize_hidden(
    hidden: Any,
    weights: dict[str, Any],
    config: dict[str, Any],
    torch: Any,
    runtime: dict[str, Any],
) -> Any:
    if not config.get("use_pre_layer_norm"):
        return hidden
    if config.get("use_rms_norm"):
        return torch_rms_norm(
            hidden,
            torch_tensor(torch, weights["final_ln_gain"], runtime),
            config["layer_norm_epsilon"],
        )
    return torch_layer_norm(
        hidden,
        torch_tensor(torch, weights["final_ln_gain"], runtime),
        torch_tensor(torch, weights["final_ln_bias"], runtime),
        config["layer_norm_epsilon"],
    )
