"""Attention-block orchestration for TinyTransformerLM."""

from __future__ import annotations

from typing import Any

from autograd import Scalar
from transformer_lm_attention import TransformerAttentionMixin
from transformer_lm_context_summaries import TransformerContextSummaryMixin
from transformer_lm_feedforward import TransformerFeedForwardMixin
from transformer_math import (
    linear_floats,
    linear_scalars,
)


class TransformerBlockMixin(
    TransformerContextSummaryMixin,
    TransformerFeedForwardMixin,
    TransformerAttentionMixin,
):
    def _forward_full_block_scalars(
        self,
        x: list[list[Scalar]],
        block: dict[str, Any],
        positions: list[int] | None = None,
    ) -> list[list[Scalar]]:
        attention_input = self._attention_input_scalars(x, block)
        q = [linear_scalars(row, block["wq"], block["bq"]) for row in attention_input]
        k = [linear_scalars(row, block["wk"], block["bk"]) for row in attention_input]
        v = [linear_scalars(row, block["wv"], block["bv"]) for row in attention_input]
        if self.config.use_rotary_positions:
            pos = positions if self.config.use_absolute_rope else None
            q = self._apply_rotary_scalars(q, pos)
            k = self._apply_rotary_scalars(k, pos)
        outputs = []
        for position in range(self.config.context_size):
            attended = self._causal_attention_scalars(q, k, v, position)
            projected = linear_scalars(attended, block["wo"], block["bo"])
            hidden = [
                x[position][dim] + projected[dim]
                for dim in range(self.config.embedding_dim)
            ]
            outputs.append(self._feed_forward_scalars(hidden, block))
        return outputs

    def _forward_final_block_floats(
        self,
        x: list[list[float]],
        block: dict[str, Any],
        context: list[int],
        positions: list[int] | None = None,
    ) -> list[float]:
        attention_input = self._attention_input_floats(x, block)
        q = [linear_floats(row, block["wq"], block["bq"]) for row in attention_input]
        k = [linear_floats(row, block["wk"], block["bk"]) for row in attention_input]
        v = [linear_floats(row, block["wv"], block["bv"]) for row in attention_input]
        if self.config.use_rotary_positions:
            pos = positions if self.config.use_absolute_rope else None
            q = self._apply_rotary_floats(q, pos)
            k = self._apply_rotary_floats(k, pos)
        last_position = self.config.context_size - 1
        attended = self._causal_attention_floats(q, k, v, last_position)
        projected = linear_floats(attended, block["wo"], block["bo"])
        hidden = [
            x[last_position][dim] + projected[dim]
            for dim in range(self.config.embedding_dim)
        ]
        hidden = self._add_final_context_summaries_floats(
            hidden,
            x,
            context,
            last_position,
        )
        return self._feed_forward_floats(hidden, block)

    def _forward_full_block_floats(
        self,
        x: list[list[float]],
        block: dict[str, Any],
        positions: list[int] | None = None,
    ) -> list[list[float]]:
        attention_input = self._attention_input_floats(x, block)
        q = [linear_floats(row, block["wq"], block["bq"]) for row in attention_input]
        k = [linear_floats(row, block["wk"], block["bk"]) for row in attention_input]
        v = [linear_floats(row, block["wv"], block["bv"]) for row in attention_input]
        if self.config.use_rotary_positions:
            pos = positions if self.config.use_absolute_rope else None
            q = self._apply_rotary_floats(q, pos)
            k = self._apply_rotary_floats(k, pos)
        outputs = []
        for position in range(self.config.context_size):
            attended = self._causal_attention_floats(q, k, v, position)
            projected = linear_floats(attended, block["wo"], block["bo"])
            hidden = [
                x[position][dim] + projected[dim]
                for dim in range(self.config.embedding_dim)
            ]
            outputs.append(self._feed_forward_floats(hidden, block))
        return outputs
