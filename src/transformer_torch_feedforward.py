"""PyTorch feed-forward helpers matching scalar transformer math."""

from __future__ import annotations

from typing import Any

from transformer_torch_norms import torch_layer_norm, torch_rms_norm
from transformer_torch_tensor_ops import torch_linear, torch_tensor


def torch_feed_forward(
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
