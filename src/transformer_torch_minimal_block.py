"""Minimal PyTorch transformer block math for parity fixtures."""

from __future__ import annotations

import math
from typing import Any

from transformer_torch_norms import torch_layer_norm, torch_rms_norm
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
    attention_input = _attention_input(x, weights, config, torch, runtime)
    q = torch.stack(
        [torch_linear(row, weights["wq"], weights["bq"], torch, runtime) for row in attention_input]
    )
    k = torch.stack(
        [torch_linear(row, weights["wk"], weights["bk"], torch, runtime) for row in attention_input]
    )
    v = torch.stack(
        [torch_linear(row, weights["wv"], weights["bv"], torch, runtime) for row in attention_input]
    )
    attended = _causal_attention(q, k, v, config, torch)
    projected = torch_linear(attended, weights["wo"], weights["bo"], torch, runtime)
    hidden = x[config["context_size"] - 1] + projected
    final_hidden = _feed_forward(hidden, weights, config, torch, runtime)
    final_hidden = _finalize_hidden(final_hidden, weights, config, torch, runtime)
    return torch_linear(final_hidden, weights["wout"], weights["bout"], torch, runtime)


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


def _feed_forward(
    hidden: Any,
    weights: dict[str, Any],
    config: dict[str, Any],
    torch: Any,
    runtime: dict[str, Any],
) -> Any:
    ff_input = _feed_forward_input(hidden, weights, config, torch, runtime)
    ff_hidden = _feed_forward_hidden(ff_input, weights, config, torch, runtime)
    ff_out = torch_linear(ff_hidden, weights["w2"], weights["b2"], torch, runtime)
    residual = hidden if config.get("use_pre_layer_norm") else ff_input
    block_out = residual + ff_out
    if config.get("use_layer_norm") and not config.get("use_pre_layer_norm"):
        return torch_layer_norm(
            block_out,
            torch_tensor(torch, weights["ln2_gain"], runtime),
            torch_tensor(torch, weights["ln2_bias"], runtime),
            config["layer_norm_epsilon"],
        )
    return block_out


def _feed_forward_hidden(
    ff_input: Any,
    weights: dict[str, Any],
    config: dict[str, Any],
    torch: Any,
    runtime: dict[str, Any],
) -> Any:
    ff_hidden = torch.tanh(
        torch_linear(ff_input, weights["w1"], weights["b1"], torch, runtime)
    )
    if not config.get("use_gated_mlp"):
        return ff_hidden
    ff_gate = torch.tanh(
        torch_linear(ff_input, weights["w_gate"], weights["b_gate"], torch, runtime)
    )
    return ff_hidden * ff_gate


def _feed_forward_input(
    hidden: Any,
    weights: dict[str, Any],
    config: dict[str, Any],
    torch: Any,
    runtime: dict[str, Any],
) -> Any:
    if config.get("use_pre_layer_norm"):
        gain_name = "ln2_gain"
        bias_name = "ln2_bias"
    else:
        gain_name = "ln1_gain"
        bias_name = "ln1_bias"
    if config.get("use_rms_norm"):
        return torch_rms_norm(
            hidden,
            torch_tensor(torch, weights[gain_name], runtime),
            config["layer_norm_epsilon"],
        )
    if config.get("use_layer_norm") or config.get("use_pre_layer_norm"):
        return torch_layer_norm(
            hidden,
            torch_tensor(torch, weights[gain_name], runtime),
            torch_tensor(torch, weights[bias_name], runtime),
            config["layer_norm_epsilon"],
        )
    return hidden


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


def _causal_attention(
    q: Any,
    k: Any,
    v: Any,
    config: dict[str, Any],
    torch: Any,
) -> Any:
    position = config["context_size"] - 1
    head_dim = config["embedding_dim"] // config["attention_heads"]
    attended = []
    for head in range(config["attention_heads"]):
        attended.extend(_attention_head(q, k, v, position, head, head_dim, torch))
    return torch.stack(attended)


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
    return [
        torch.stack(
            [weights[past] * v[past][dim] for past in range(position + 1)]
        ).sum()
        for dim in range(start, end)
    ]
