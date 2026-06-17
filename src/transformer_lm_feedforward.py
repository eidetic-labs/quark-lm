"""Feed-forward and block input normalization helpers for TinyTransformerLM."""

from __future__ import annotations

import math
from typing import Any

from autograd import Scalar
from transformer_math import (
    layer_norm_floats,
    layer_norm_scalars,
    linear_floats,
    linear_scalars,
    rms_norm_floats,
    rms_norm_scalars,
)


class TransformerFeedForwardMixin:
    def _feed_forward_scalars(
        self,
        hidden: list[Scalar],
        block: dict[str, Any],
    ) -> list[Scalar]:
        if self.config.use_pre_layer_norm:
            ff_input = layer_norm_scalars(
                hidden,
                block["ln2_gain"],
                block["ln2_bias"],
                self.config.layer_norm_epsilon,
            )
            if self.config.use_rms_norm:
                ff_input = rms_norm_scalars(
                    hidden,
                    block["ln2_gain"],
                    self.config.layer_norm_epsilon,
                )
            ff_hidden = [
                value.tanh()
                for value in linear_scalars(ff_input, block["w1"], block["b1"])
            ]
            if self.config.use_gated_mlp:
                ff_gate = [
                    value.tanh()
                    for value in linear_scalars(ff_input, block["w_gate"], block["b_gate"])
                ]
                ff_hidden = [
                    hidden_value * gate_value
                    for hidden_value, gate_value in zip(ff_hidden, ff_gate)
                ]
            ff_out = linear_scalars(ff_hidden, block["w2"], block["b2"])
            return [
                hidden[dim] + ff_out[dim]
                for dim in range(self.config.embedding_dim)
            ]
        if self.config.use_layer_norm:
            hidden = layer_norm_scalars(
                hidden,
                block["ln1_gain"],
                block["ln1_bias"],
                self.config.layer_norm_epsilon,
            )
        if self.config.use_rms_norm:
            hidden = rms_norm_scalars(
                hidden,
                block["ln1_gain"],
                self.config.layer_norm_epsilon,
            )
        ff_hidden = [
            value.tanh()
            for value in linear_scalars(hidden, block["w1"], block["b1"])
        ]
        if self.config.use_gated_mlp:
            ff_gate = [
                value.tanh()
                for value in linear_scalars(hidden, block["w_gate"], block["b_gate"])
            ]
            ff_hidden = [
                hidden_value * gate_value
                for hidden_value, gate_value in zip(ff_hidden, ff_gate)
            ]
        ff_out = linear_scalars(ff_hidden, block["w2"], block["b2"])
        block_out = [
            hidden[dim] + ff_out[dim]
            for dim in range(self.config.embedding_dim)
        ]
        if self.config.use_layer_norm:
            block_out = layer_norm_scalars(
                block_out,
                block["ln2_gain"],
                block["ln2_bias"],
                self.config.layer_norm_epsilon,
            )
        return block_out

    def _attention_input_scalars(
        self,
        x: list[list[Scalar]],
        block: dict[str, Any],
    ) -> list[list[Scalar]]:
        if self.config.use_rms_norm and self.config.use_pre_layer_norm:
            return [
                rms_norm_scalars(
                    row,
                    block["ln1_gain"],
                    self.config.layer_norm_epsilon,
                )
                for row in x
            ]
        if not self.config.use_pre_layer_norm:
            return x
        return [
            layer_norm_scalars(
                row,
                block["ln1_gain"],
                block["ln1_bias"],
                self.config.layer_norm_epsilon,
            )
            for row in x
        ]

    def _feed_forward_floats(
        self,
        hidden: list[float],
        block: dict[str, Any],
    ) -> list[float]:
        if self.config.use_pre_layer_norm:
            ff_input = layer_norm_floats(
                hidden,
                block["ln2_gain"],
                block["ln2_bias"],
                self.config.layer_norm_epsilon,
            )
            if self.config.use_rms_norm:
                ff_input = rms_norm_floats(
                    hidden,
                    block["ln2_gain"],
                    self.config.layer_norm_epsilon,
                )
            ff_hidden = [
                math.tanh(value)
                for value in linear_floats(ff_input, block["w1"], block["b1"])
            ]
            if self.config.use_gated_mlp:
                ff_gate = [
                    math.tanh(value)
                    for value in linear_floats(ff_input, block["w_gate"], block["b_gate"])
                ]
                ff_hidden = [
                    hidden_value * gate_value
                    for hidden_value, gate_value in zip(ff_hidden, ff_gate)
                ]
            ff_out = linear_floats(ff_hidden, block["w2"], block["b2"])
            return [
                hidden[dim] + ff_out[dim]
                for dim in range(self.config.embedding_dim)
            ]
        if self.config.use_layer_norm:
            hidden = layer_norm_floats(
                hidden,
                block["ln1_gain"],
                block["ln1_bias"],
                self.config.layer_norm_epsilon,
            )
        if self.config.use_rms_norm:
            hidden = rms_norm_floats(
                hidden,
                block["ln1_gain"],
                self.config.layer_norm_epsilon,
            )
        ff_hidden = [
            math.tanh(value)
            for value in linear_floats(hidden, block["w1"], block["b1"])
        ]
        if self.config.use_gated_mlp:
            ff_gate = [
                math.tanh(value)
                for value in linear_floats(hidden, block["w_gate"], block["b_gate"])
            ]
            ff_hidden = [
                hidden_value * gate_value
                for hidden_value, gate_value in zip(ff_hidden, ff_gate)
            ]
        ff_out = linear_floats(ff_hidden, block["w2"], block["b2"])
        block_out = [
            hidden[dim] + ff_out[dim]
            for dim in range(self.config.embedding_dim)
        ]
        if self.config.use_layer_norm:
            block_out = layer_norm_floats(
                block_out,
                block["ln2_gain"],
                block["ln2_bias"],
                self.config.layer_norm_epsilon,
            )
        return block_out

    def _attention_input_floats(
        self,
        x: list[list[float]],
        block: dict[str, Any],
    ) -> list[list[float]]:
        if self.config.use_rms_norm and self.config.use_pre_layer_norm:
            return [
                rms_norm_floats(
                    row,
                    block["ln1_gain"],
                    self.config.layer_norm_epsilon,
                )
                for row in x
            ]
        if not self.config.use_pre_layer_norm:
            return x
        return [
            layer_norm_floats(
                row,
                block["ln1_gain"],
                block["ln1_bias"],
                self.config.layer_norm_epsilon,
            )
            for row in x
        ]
