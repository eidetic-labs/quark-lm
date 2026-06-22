"""Tier-2 tensorized (B,C,D) PyTorch forward with masked causal attention.

Opt-in batched mirror of ``torch_minimal_logits`` (the per-position Tier-1 path).
Runs only behind ``runtime['use_batched_forward']`` AND a clean
``batched_forward_unsupported_reason`` gate, so the default per-position path, the
scalar engine, and the fake-torch double are untouched. The math is the same
softmax((q.k)*scale).v / norms / feed-forward as Tier-1; reductions reorder
(SDPA + dim=-1 reductions), so it holds to the validated parity band, not bit-exact.
"""

from __future__ import annotations

import math
from typing import Any

from transformer_torch_tensor_ops import torch_linear, torch_tensor


def torch_batched_logits(
    contexts: list[list[int]],
    fixture: dict[str, Any],
    torch: Any,
    runtime: dict[str, Any],
) -> Any:
    """Logits (B, vocab) for a batch of contexts on the same weights."""

    weights = fixture["weights"]
    config = fixture["model_config"]
    token_embeddings = torch_tensor(torch, weights["token_embeddings"], runtime)
    position_embeddings = torch_tensor(torch, weights["position_embeddings"], runtime)
    context_size = config["context_size"]
    idx = torch.tensor(contexts, dtype=torch.long, device=runtime["device"])
    x = token_embeddings[idx] + position_embeddings[:context_size].unsqueeze(0)
    blocks = [weights] + list(weights.get("extra_layers", []))
    for block in blocks[:-1]:
        x = _forward_full_block(x, block, config, torch, runtime)
    final_hidden = _forward_final_block(
        x, idx, blocks[-1], weights, config, torch, runtime
    )
    final_hidden = _finalize_hidden(final_hidden, weights, config, torch, runtime)
    output_weights = weights["wout"]
    if config.get("tie_output_embeddings"):
        output_weights = token_embeddings.transpose(0, 1)
    return torch_linear(final_hidden, output_weights, weights["bout"], torch, runtime)


_COMPILED_BATCHED_LOGITS = None


def batched_logits_fn(torch: Any, runtime: dict[str, Any]) -> Any:
    """``torch_batched_logits``, optionally torch.compile'd via runtime['use_compile'].

    Device-agnostic: torch.compile targets the CUDA/MPS/CPU backend transparently,
    and the compiled graph is built once and reused. Default off returns the eager
    function unchanged. Only the tensorized batched path is compiled (the
    per-position Python builder graph-breaks with no benefit); since the batched
    path is never reached under the fake-torch double, torch.compile is never
    invoked there.
    """

    if not runtime.get("use_compile"):
        return torch_batched_logits
    global _COMPILED_BATCHED_LOGITS
    if _COMPILED_BATCHED_LOGITS is None:
        _COMPILED_BATCHED_LOGITS = torch.compile(torch_batched_logits)
    return _COMPILED_BATCHED_LOGITS


def _forward_full_block(x, block, config, torch, runtime):
    attended = _attention(x, block, config, torch, runtime)
    return _feed_forward(attended, block, config, torch, runtime)


def _forward_final_block(x, idx, block, weights, config, torch, runtime):
    attended = _attention(x, block, config, torch, runtime)
    last = attended[:, config["context_size"] - 1, :]
    last = _add_context_summaries(last, x, idx, weights, config, torch, runtime)
    return _feed_forward(last, block, config, torch, runtime)


def _attention(x, block, config, torch, runtime):
    attention_input = _attention_input(x, block, config, torch, runtime)
    q = torch_linear(attention_input, block["wq"], block["bq"], torch, runtime)
    k = torch_linear(attention_input, block["wk"], block["bk"], torch, runtime)
    v = torch_linear(attention_input, block["wv"], block["bv"], torch, runtime)
    head_dim = config["embedding_dim"] // config["attention_heads"]
    if config.get("use_rotary_positions"):
        q = _apply_rotary_batched(q, config, head_dim, torch, runtime)
        k = _apply_rotary_batched(k, config, head_dim, torch, runtime)
    attended = _sdpa(q, k, v, config, head_dim, torch)
    projected = torch_linear(attended, block["wo"], block["bo"], torch, runtime)
    return x + projected


def _sdpa(q, k, v, config, head_dim, torch):
    batch, context_size = q.shape[0], q.shape[1]
    heads = config["attention_heads"]

    def split(t):
        return t.reshape(batch, context_size, heads, head_dim).transpose(1, 2)

    attended = torch.nn.functional.scaled_dot_product_attention(
        split(q), split(k), split(v), is_causal=True, scale=1.0 / math.sqrt(head_dim)
    )
    return attended.transpose(1, 2).reshape(batch, context_size, heads * head_dim)


def _apply_rotary_batched(rows, config, head_dim, torch, runtime):
    """RoPE reproducing the exact head*head_dim+offset addressing of Tier-1.

    Pairs (index, index+1) WITHIN each head block via ``range(0, head_dim - 1, 2)``;
    the odd-head_dim tail dim is left untouched (matches ``_apply_rotary_row``). NOT a
    naive ::2/1::2 split, which would cross head boundaries and mishandle odd head_dim.
    """

    context_size = rows.shape[1]
    positions = torch.arange(context_size, dtype=rows.dtype, device=runtime["device"])
    output = rows.clone()
    for head in range(config["attention_heads"]):
        start = head * head_dim
        for offset in range(0, head_dim - 1, 2):
            index = start + offset
            angle = positions / (10000.0 ** (offset / max(head_dim, 1)))
            cos = angle.cos().unsqueeze(0).unsqueeze(-1)
            sin = angle.sin().unsqueeze(0).unsqueeze(-1)
            left = rows[:, :, index : index + 1]
            right = rows[:, :, index + 1 : index + 2]
            output[:, :, index : index + 1] = left * cos - right * sin
            output[:, :, index + 1 : index + 2] = left * sin + right * cos
    return output


def _feed_forward(hidden, weights, config, torch, runtime):
    ff_input = _feed_forward_input(hidden, weights, config, torch, runtime)
    ff_hidden = torch.tanh(torch_linear(ff_input, weights["w1"], weights["b1"], torch, runtime))
    if config.get("use_gated_mlp"):
        gate = torch.tanh(
            torch_linear(ff_input, weights["w_gate"], weights["b_gate"], torch, runtime)
        )
        ff_hidden = ff_hidden * gate
    ff_out = torch_linear(ff_hidden, weights["w2"], weights["b2"], torch, runtime)
    residual = hidden if config.get("use_pre_layer_norm") else ff_input
    block_out = residual + ff_out
    if config.get("use_layer_norm") and not config.get("use_pre_layer_norm"):
        return _layer_norm(
            block_out,
            torch_tensor(torch, weights["ln2_gain"], runtime),
            torch_tensor(torch, weights["ln2_bias"], runtime),
            config["layer_norm_epsilon"],
        )
    return block_out


def _attention_input(x, weights, config, torch, runtime):
    if not config.get("use_pre_layer_norm"):
        return x
    gain = torch_tensor(torch, weights["ln1_gain"], runtime)
    if config.get("use_rms_norm"):
        return _rms_norm(x, gain, config["layer_norm_epsilon"])
    bias = torch_tensor(torch, weights["ln1_bias"], runtime)
    return _layer_norm(x, gain, bias, config["layer_norm_epsilon"])


def _feed_forward_input(hidden, weights, config, torch, runtime):
    gain_name = "ln2_gain" if config.get("use_pre_layer_norm") else "ln1_gain"
    bias_name = "ln2_bias" if config.get("use_pre_layer_norm") else "ln1_bias"
    if config.get("use_rms_norm"):
        return _rms_norm(
            hidden, torch_tensor(torch, weights[gain_name], runtime), config["layer_norm_epsilon"]
        )
    if config.get("use_layer_norm") or config.get("use_pre_layer_norm"):
        return _layer_norm(
            hidden,
            torch_tensor(torch, weights[gain_name], runtime),
            torch_tensor(torch, weights[bias_name], runtime),
            config["layer_norm_epsilon"],
        )
    return hidden


def _finalize_hidden(hidden, weights, config, torch, runtime):
    if not config.get("use_pre_layer_norm"):
        return hidden
    gain = torch_tensor(torch, weights["final_ln_gain"], runtime)
    if config.get("use_rms_norm"):
        return _rms_norm(hidden, gain, config["layer_norm_epsilon"])
    bias = torch_tensor(torch, weights["final_ln_bias"], runtime)
    return _layer_norm(hidden, gain, bias, config["layer_norm_epsilon"])


def _layer_norm(values, gain, bias, epsilon):
    mean = values.mean(dim=-1, keepdim=True)
    centered = values - mean
    variance = (centered * centered).mean(dim=-1, keepdim=True)
    return centered * (variance + epsilon).pow(-0.5) * gain + bias


def _rms_norm(values, gain, epsilon):
    mean_square = (values * values).mean(dim=-1, keepdim=True)
    return values * (mean_square + epsilon).pow(-0.5) * gain


def _add_context_summaries(last, x, idx, weights, config, torch, runtime):
    """Position-collapsing summaries on the (B, D) last-position slice.

    use_prompt_position_projection is deliberately NOT handled here -- it is
    position-indexed and a collapsing reduction would lose the per-position W[p].
    batched_forward_unsupported_reason fails that profile closed to Tier-1.
    """

    from transformer_torch_batched_summaries import add_batched_context_summaries

    return add_batched_context_summaries(
        last, x, idx, weights, config, torch, runtime
    )
