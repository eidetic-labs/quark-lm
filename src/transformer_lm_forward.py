"""Forward-pass orchestration for TinyTransformerLM."""

from __future__ import annotations

from typing import Any

from autograd import Scalar
from transformer_lm_context_summaries import TransformerContextSummaryMixin
from transformer_math import (
    layer_norm_floats,
    layer_norm_scalars,
    linear_floats,
    linear_scalars,
    matrix_to_floats,
    matrix_to_scalars,
    rms_norm_floats,
    rms_norm_scalars,
    vector_to_floats,
)


class TransformerForwardMixin(TransformerContextSummaryMixin):
    def _forward_scalars(
        self, context: list[int], positions: list[int] | None = None
    ) -> list[Scalar]:
        return linear_scalars(
            self._final_hidden_scalars(context, positions),
            self._output_weights_scalars(),
            self.bout,
        )

    def _final_hidden_scalars(
        self, context: list[int], positions: list[int] | None = None
    ) -> list[Scalar]:
        if len(context) != self.config.context_size:
            raise ValueError(
                f"context must have {self.config.context_size} ids, got {len(context)}"
            )
        # Phase 2: under ``use_absolute_rope`` the LEARNED position_embeddings addend is
        # dropped (RoPE becomes the sole positional source); the OFF arm is the verbatim
        # pre-Phase-2 expression so flag-off stays byte-exact. Structural two-branch
        # if/else (NOT ``+ 0.0``, which would add a phantom Scalar(0.0) autograd node and
        # could perturb the f64 1e-6 band). ``positions`` is the absolute stream index per
        # slot, threaded into the blocks where the RoPE site consumes it under the flag.
        add_pos = not self.config.use_absolute_rope
        if self.freeze_lower_layers_for_updates and self.config.num_layers > 1:
            float_blocks = [self._block_to_floats(block) for block in self.blocks[:-1]]
            token_embeddings = matrix_to_floats(self.token_embeddings)
            position_embeddings = matrix_to_floats(self.position_embeddings)
            if add_pos:
                x_float = [
                    [
                        token_embeddings[token_id][dim] + position_embeddings[position][dim]
                        for dim in range(self.config.embedding_dim)
                    ]
                    for position, token_id in enumerate(context)
                ]
            else:
                x_float = [
                    [
                        token_embeddings[token_id][dim]
                        for dim in range(self.config.embedding_dim)
                    ]
                    for position, token_id in enumerate(context)
                ]
            for block in float_blocks:
                x_float = self._forward_full_block_floats(x_float, block, positions)
            x = matrix_to_scalars(x_float)
            return self._finalize_hidden_scalars(
                self._forward_final_block_scalars(x, self.blocks[-1], context, positions)
            )
        if add_pos:
            x = [
                [
                    self.token_embeddings[token_id][dim] + self.position_embeddings[position][dim]
                    for dim in range(self.config.embedding_dim)
                ]
                for position, token_id in enumerate(context)
            ]
        else:
            x = [
                [
                    self.token_embeddings[token_id][dim]
                    for dim in range(self.config.embedding_dim)
                ]
                for position, token_id in enumerate(context)
            ]
        if self.config.num_layers == 1:
            return self._finalize_hidden_scalars(
                self._forward_final_block_scalars(x, self.blocks[0], context, positions)
            )
        else:
            for block in self.blocks[:-1]:
                x = self._forward_full_block_scalars(x, block, positions)
            return self._finalize_hidden_scalars(
                self._forward_final_block_scalars(x, self.blocks[-1], context, positions)
            )

    def _forward_floats(
        self, context: list[int], positions: list[int] | None = None
    ) -> list[float]:
        return linear_floats(
            self.final_hidden(context, positions),
            self._output_weights_floats(),
            vector_to_floats(self.bout),
        )

    def final_hidden(
        self, context: list[int], positions: list[int] | None = None
    ) -> list[float]:
        if len(context) != self.config.context_size:
            raise ValueError(
                f"context must have {self.config.context_size} ids, got {len(context)}"
            )
        token_embeddings = matrix_to_floats(self.token_embeddings)
        position_embeddings = matrix_to_floats(self.position_embeddings)
        # Phase 2: drop the learned pos-embed addend under use_absolute_rope (RoPE is the
        # sole positional source); OFF arm is the verbatim original expression.
        add_pos = not self.config.use_absolute_rope
        if add_pos:
            x = [
                [
                    token_embeddings[token_id][dim] + position_embeddings[position][dim]
                    for dim in range(self.config.embedding_dim)
                ]
                for position, token_id in enumerate(context)
            ]
        else:
            x = [
                [
                    token_embeddings[token_id][dim]
                    for dim in range(self.config.embedding_dim)
                ]
                for position, token_id in enumerate(context)
            ]
        float_blocks = [self._block_to_floats(block) for block in self.blocks]
        if self.config.num_layers == 1:
            return self._finalize_hidden_floats(
                self._forward_final_block_floats(x, float_blocks[0], context, positions)
            )
        else:
            for block in float_blocks[:-1]:
                x = self._forward_full_block_floats(x, block, positions)
            return self._finalize_hidden_floats(
                self._forward_final_block_floats(x, float_blocks[-1], context, positions)
            )

    def _finalize_hidden_scalars(self, hidden: list[Scalar]) -> list[Scalar]:
        if not self.config.use_pre_layer_norm:
            return hidden
        if self.config.use_rms_norm:
            return rms_norm_scalars(
                hidden,
                self.final_ln_gain,
                self.config.layer_norm_epsilon,
            )
        return layer_norm_scalars(
            hidden,
            self.final_ln_gain,
            self.final_ln_bias,
            self.config.layer_norm_epsilon,
        )

    def _finalize_hidden_floats(self, hidden: list[float]) -> list[float]:
        if not self.config.use_pre_layer_norm:
            return hidden
        if self.config.use_rms_norm:
            return rms_norm_floats(
                hidden,
                vector_to_floats(self.final_ln_gain),
                self.config.layer_norm_epsilon,
            )
        return layer_norm_floats(
            hidden,
            vector_to_floats(self.final_ln_gain),
            vector_to_floats(self.final_ln_bias),
            self.config.layer_norm_epsilon,
        )

    def _forward_final_block_scalars(
        self,
        x: list[list[Scalar]],
        block: dict[str, Any],
        context: list[int],
        positions: list[int] | None = None,
    ) -> list[Scalar]:
        attention_input = self._attention_input_scalars(x, block)
        q = [linear_scalars(row, block["wq"], block["bq"]) for row in attention_input]
        k = [linear_scalars(row, block["wk"], block["bk"]) for row in attention_input]
        v = [linear_scalars(row, block["wv"], block["bv"]) for row in attention_input]
        if self.config.use_rotary_positions:
            pos = positions if self.config.use_absolute_rope else None
            q = self._apply_rotary_scalars(q, pos)
            k = self._apply_rotary_scalars(k, pos)
        last_position = self.config.context_size - 1
        attended = self._causal_attention_scalars(q, k, v, last_position)
        projected = linear_scalars(attended, block["wo"], block["bo"])
        hidden = [
            x[last_position][dim] + projected[dim]
            for dim in range(self.config.embedding_dim)
        ]
        hidden = self._add_final_context_summaries_scalars(
            hidden,
            x,
            context,
            last_position,
        )
        return self._feed_forward_scalars(hidden, block)
