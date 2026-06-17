"""Parameter view construction for TinyTransformerLM."""

from __future__ import annotations

from typing import Any

from autograd import Scalar
from transformer_math import flatten_scalars
from transformer_model import TransformerConfig


def uses_block_layer_norm_parameters(config: TransformerConfig) -> bool:
    return config.use_layer_norm or config.use_pre_layer_norm


def all_transformer_parameters(model: Any) -> list[Scalar]:
    params: list[Scalar] = []
    for item in [
        model.token_embeddings,
        model.position_embeddings,
        model.wq,
        model.bq,
        model.wk,
        model.bk,
        model.wv,
        model.bv,
        model.wo,
        model.bo,
        model.w1,
        model.b1,
        model.w2,
        model.b2,
        model.bout,
    ]:
        params.extend(flatten_scalars(item))
    _extend_optional_global_parameters(model, params)
    if uses_block_layer_norm_parameters(model.config):
        for item in [model.ln1_gain, model.ln1_bias, model.ln2_gain, model.ln2_bias]:
            params.extend(flatten_scalars(item))
    if model.config.use_pre_layer_norm:
        for item in [model.final_ln_gain, model.final_ln_bias]:
            params.extend(flatten_scalars(item))
    for block in model.extra_blocks:
        _extend_block_parameters(model, block, params)
    return params


def top_layer_transformer_parameters(model: Any) -> list[Scalar]:
    if model.config.num_layers == 1:
        return all_transformer_parameters(model)
    params: list[Scalar] = []
    top_block = model.blocks[-1]
    for item in [
        top_block["wq"],
        top_block["bq"],
        top_block["wk"],
        top_block["bk"],
        top_block["wv"],
        top_block["bv"],
        top_block["wo"],
        top_block["bo"],
        top_block["w1"],
        top_block["b1"],
        top_block["w2"],
        top_block["b2"],
        model.bout,
    ]:
        params.extend(flatten_scalars(item))
    if model.config.use_gated_mlp:
        for item in [top_block["w_gate"], top_block["b_gate"]]:
            params.extend(flatten_scalars(item))
    _extend_optional_global_parameters(model, params, include_gated_mlp=False)
    if uses_block_layer_norm_parameters(model.config):
        for item in [
            top_block["ln1_gain"],
            top_block["ln1_bias"],
            top_block["ln2_gain"],
            top_block["ln2_bias"],
        ]:
            params.extend(flatten_scalars(item))
    if model.config.use_pre_layer_norm:
        for item in [model.final_ln_gain, model.final_ln_bias]:
            params.extend(flatten_scalars(item))
    return params


def _extend_optional_global_parameters(
    model: Any,
    params: list[Scalar],
    *,
    include_gated_mlp: bool = True,
) -> None:
    if include_gated_mlp and model.config.use_gated_mlp:
        for item in [model.w_gate, model.b_gate]:
            params.extend(flatten_scalars(item))
    if not model.config.tie_output_embeddings:
        params.extend(flatten_scalars(model.wout))
    if model.config.use_context_projection:
        for item in [model.context_projection_w, model.context_projection_b]:
            params.extend(flatten_scalars(item))
    if model.config.use_prompt_prefix_projection:
        for item in [
            model.prompt_prefix_projection_w,
            model.prompt_prefix_projection_b,
        ]:
            params.extend(flatten_scalars(item))
    if model.config.use_prompt_position_projection:
        for item in [
            model.prompt_position_projection_w,
            model.prompt_position_projection_b,
        ]:
            params.extend(flatten_scalars(item))
    if model.config.use_prompt_attention_summary:
        for item in [
            model.prompt_summary_query,
            model.prompt_summary_w,
            model.prompt_summary_b,
        ]:
            params.extend(flatten_scalars(item))


def _extend_block_parameters(
    model: Any,
    block: dict[str, Any],
    params: list[Scalar],
) -> None:
    for item in [
        block["wq"],
        block["bq"],
        block["wk"],
        block["bk"],
        block["wv"],
        block["bv"],
        block["wo"],
        block["bo"],
        block["w1"],
        block["b1"],
        block["w2"],
        block["b2"],
    ]:
        params.extend(flatten_scalars(item))
    if model.config.use_gated_mlp:
        for item in [block["w_gate"], block["b_gate"]]:
            params.extend(flatten_scalars(item))
    if uses_block_layer_norm_parameters(model.config):
        for item in [
            block["ln1_gain"],
            block["ln1_bias"],
            block["ln2_gain"],
            block["ln2_bias"],
        ]:
            params.extend(flatten_scalars(item))
