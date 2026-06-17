"""Float final-context summary helpers for TinyTransformerLM."""

from __future__ import annotations

import math

from transformer_math import (
    dot_floats,
    linear_floats,
    matrix_to_floats,
    softmax_floats,
    vector_to_floats,
)


class TransformerFloatContextSummaryMixin:
    def _add_final_context_summaries_floats(
        self,
        hidden: list[float],
        x: list[list[float]],
        context: list[int],
        last_position: int,
    ) -> list[float]:
        if self.config.use_context_mean:
            hidden = self._add_context_mean_floats(hidden, x)
        if self.config.use_context_projection:
            hidden = self._add_context_projection_floats(hidden, x)
        if self.config.use_prompt_prefix_projection:
            hidden = self._add_prompt_prefix_projection_floats(
                hidden,
                x,
                context,
                last_position,
            )
        if self.config.use_prompt_position_projection:
            hidden = self._add_prompt_position_projection_floats(
                hidden,
                x,
                context,
                last_position,
            )
        if self.config.use_prompt_attention_summary:
            hidden = self._add_prompt_attention_summary_floats(hidden, x)
        return hidden

    def _add_context_mean_floats(
        self,
        hidden: list[float],
        x: list[list[float]],
    ) -> list[float]:
        return [
            hidden[dim] + sum(row[dim] for row in x) / self.config.context_size
            for dim in range(self.config.embedding_dim)
        ]

    def _add_context_projection_floats(
        self,
        hidden: list[float],
        x: list[list[float]],
    ) -> list[float]:
        context_summary = [
            sum(row[dim] for row in x) / self.config.context_size
            for dim in range(self.config.embedding_dim)
        ]
        projected_summary = linear_floats(
            context_summary,
            matrix_to_floats(self.context_projection_w),
            vector_to_floats(self.context_projection_b),
        )
        return [
            hidden[dim] + projected_summary[dim]
            for dim in range(self.config.embedding_dim)
        ]

    def _add_prompt_prefix_projection_floats(
        self,
        hidden: list[float],
        x: list[list[float]],
        context: list[int],
        last_position: int,
    ) -> list[float]:
        prompt_rows = [
            row
            for position, row in enumerate(x[:last_position])
            if context[position] != 0
        ]
        if not prompt_rows:
            return hidden
        prompt_summary = [
            sum(row[dim] for row in prompt_rows) / len(prompt_rows)
            for dim in range(self.config.embedding_dim)
        ]
        projected_summary = linear_floats(
            prompt_summary,
            matrix_to_floats(self.prompt_prefix_projection_w),
            vector_to_floats(self.prompt_prefix_projection_b),
        )
        return [
            hidden[dim] + projected_summary[dim]
            for dim in range(self.config.embedding_dim)
        ]

    def _add_prompt_position_projection_floats(
        self,
        hidden: list[float],
        x: list[list[float]],
        context: list[int],
        last_position: int,
    ) -> list[float]:
        prompt_positions = [
            (position, row)
            for position, row in enumerate(x[:last_position])
            if context[position] != 0
        ]
        if not prompt_positions:
            return hidden
        prompt_position_projection_w = [
            matrix_to_floats(position_weights)
            for position_weights in self.prompt_position_projection_w
        ]
        prompt_position_projection_b = vector_to_floats(
            self.prompt_position_projection_b
        )
        projected_summary = []
        for output_dim, bias in enumerate(prompt_position_projection_b):
            total = 0.0
            for position, row in prompt_positions:
                position_weights = prompt_position_projection_w[position]
                for input_dim, value in enumerate(row):
                    total += value * position_weights[input_dim][output_dim]
            projected_summary.append(total / len(prompt_positions) + bias)
        return [
            hidden[dim]
            + projected_summary[dim] * self.config.prompt_position_projection_scale
            for dim in range(self.config.embedding_dim)
        ]

    def _add_prompt_attention_summary_floats(
        self,
        hidden: list[float],
        x: list[list[float]],
    ) -> list[float]:
        prompt_summary_query = vector_to_floats(self.prompt_summary_query)
        scores = [
            dot_floats(prompt_summary_query, row)
            * (1.0 / math.sqrt(self.config.embedding_dim))
            for row in x
        ]
        weights = softmax_floats(scores)
        attention_summary = [
            sum(weight * row[dim] for row, weight in zip(x, weights))
            for dim in range(self.config.embedding_dim)
        ]
        projected_summary = linear_floats(
            attention_summary,
            matrix_to_floats(self.prompt_summary_w),
            vector_to_floats(self.prompt_summary_b),
        )
        return [
            hidden[dim] + projected_summary[dim]
            for dim in range(self.config.embedding_dim)
        ]
