"""Minimal PyTorch transformer block math for parity fixtures."""

from __future__ import annotations

from typing import Any

from transformer_torch_attention import (
    _apply_rotary_row,
    torch_apply_rotary,
    torch_causal_attention,
)
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
    # Phase 2: under use_absolute_rope the learned position_embeddings addend is dropped
    # (RoPE is the sole positional source); the OFF arm is the verbatim pre-Phase-2
    # expression so flag-off is byte-identical. The position_embeddings tensor is still
    # materialized (minimal diff + format stability), just unread under the flag.
    if config.get("use_absolute_rope"):
        x = torch.stack([token_embeddings[token_id] for token_id in context])
    else:
        x = torch.stack(
            [
                token_embeddings[token_id] + position_embeddings[position]
                for position, token_id in enumerate(context)
            ]
        )
    blocks = _transformer_blocks(weights)
    for layer_index, block in enumerate(blocks[:-1]):
        # Layer 0 (layer_index 0) is cacheable; upper layers recompute K/V every step.
        x = _forward_full_block(
            x, block, config, torch, runtime, layer_index=layer_index
        )
    final_hidden = _forward_final_block(
        x,
        context,
        blocks[-1],
        weights,
        config,
        torch,
        runtime,
        # The final block is layer 0 ONLY when it is the sole block (1-layer model);
        # otherwise it is an upper layer and must recompute.
        layer_index=len(blocks) - 1,
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
    layer_index: int = 0,
) -> Any:
    q, k, v = _attention_projections(
        x, block, config, torch, runtime, layer_index=layer_index
    )
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
    layer_index: int = 0,
) -> Any:
    q, k, v = _attention_projections(
        x, block, config, torch, runtime, layer_index=layer_index
    )
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
    layer_index: int = 0,
) -> tuple[Any, Any, Any]:
    attention_input = _attention_input(x, block, config, torch, runtime)
    q = _project_rows(attention_input, block, "wq", "bq", torch, runtime)
    if config.get("use_rotary_positions"):
        # Consumption gated on use_absolute_rope: absent the flag, positions are
        # dropped -> slot-keyed (enumerate), byte-identical to the pre-absolute path.
        # Phase 2 FAIL-CLOSED guard (R-C): under the flag the pos-embed addend is gone,
        # so RoPE is the ONLY positional signal -- a missing abs_positions would silently
        # slot-key (enumerate on None at attention.py), wasting a retrain. Crash instead,
        # so any training site that failed to thread positions is caught loudly.
        if config.get("use_absolute_rope"):
            positions = runtime.get("abs_positions")
            if positions is None:
                raise ValueError(
                    "use_absolute_rope set but runtime['abs_positions'] missing -> "
                    "would silently slot-key; training/forward site failed to thread "
                    "positions"
                )
        else:
            positions = None
        q = torch_apply_rotary(q, config, torch, positions)
    else:
        positions = None
    k, v = _layer0_kv(
        attention_input, block, config, torch, runtime, positions, layer_index
    )
    return q, k, v


def _layer0_kv(
    attention_input: Any,
    block: dict[str, Any],
    config: dict[str, Any],
    torch: Any,
    runtime: dict[str, Any],
    positions: list[int] | None,
    layer_index: int,
) -> tuple[Any, Any]:
    """Torch mirror of TinyTransformerLM._layer0_kv_floats: K/V for the block.

    REGIME GATE + LAYER-0-ONLY: the layer-0 append-valid KV cache (runtime['kv_cache'])
    is consulted ONLY for layer 0 (layer_index == 0) AND only under the write-once
    geometry (use_absolute_rope + use_rotary_positions). Otherwise -- upper layers, any
    non-thesis geometry, or no cache -- K/V is fully recomputed, byte-identical to the
    prior path. In-regime, only the newest token's rotated-K / raw-V is computed; the
    historical rows are served from the cache in ascending-slot order so the value
    aggregation order matches the recompute.
    """

    rope_on = bool(config.get("use_rotary_positions"))
    write_once_regime = rope_on and bool(config.get("use_absolute_rope"))
    cache = runtime.get("kv_cache") if (layer_index == 0 and write_once_regime) else None

    def project_k(rows: Any) -> Any:
        return _project_rows(rows, block, "wk", "bk", torch, runtime)

    if cache is None:
        k = project_k(attention_input)
        v = _project_rows(attention_input, block, "wv", "bv", torch, runtime)
        if rope_on:
            k = torch_apply_rotary(k, config, torch, positions)
        return k, v

    context_size = config["context_size"]
    last_position = context_size - 1
    slot_positions = (
        positions if positions is not None else list(range(context_size))
    )

    def compute_row(slot_index: int) -> tuple[Any, Any]:
        row = attention_input[slot_index]
        key_row = torch_linear(row, block["wk"], block["bk"], torch, runtime)
        value_row = torch_linear(row, block["wv"], block["bv"], torch, runtime)
        if rope_on:
            angle_pos = (
                positions[slot_index] if positions is not None else slot_index
            )
            key_row = _apply_rotary_row(key_row, angle_pos, config, torch)
        return key_row, value_row

    # Compute + store the newest token (the last slot); a pad position is skipped.
    new_key, new_value = compute_row(last_position)
    cache.store(slot_positions[last_position], new_key, new_value)

    def assemble_row(slot_index: int) -> tuple[Any, Any]:
        if slot_index == last_position:
            return new_key, new_value
        return compute_row(slot_index)

    return cache.assemble(slot_positions, assemble_row, torch)


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
