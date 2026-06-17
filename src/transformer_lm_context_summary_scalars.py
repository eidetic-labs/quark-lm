"""Scalar final-context summary helpers for TinyTransformerLM."""

from __future__ import annotations

import math

from autograd import Scalar
from transformer_math import dot_scalars, linear_scalars, softmax_scalars


class TransformerScalarContextSummaryMixin:
    def _add_final_context_summaries_scalars(
        self,
        hidden: list[Scalar],
        x: list[list[Scalar]],
        context: list[int],
        last_position: int,
    ) -> list[Scalar]:
        if self.config.use_context_mean:
            hidden = self._add_context_mean_scalars(hidden, x)
        if self.config.use_context_projection:
            hidden = self._add_context_projection_scalars(hidden, x)
        if self.config.use_prompt_prefix_projection:
            hidden = self._add_prompt_prefix_projection_scalars(
                hidden,
                x,
                context,
                last_position,
            )
        if self.config.use_prompt_position_projection:
            hidden = self._add_prompt_position_projection_scalars(
                hidden,
                x,
                context,
                last_position,
            )
        if self.config.use_prompt_attention_summary:
            hidden = self._add_prompt_attention_summary_scalars(hidden, x)
        return hidden

    def _add_context_mean_scalars(
        self,
        hidden: list[Scalar],
        x: list[list[Scalar]],
    ) -> list[Scalar]:
        return [
            hidden[dim] + sum(row[dim] for row in x) / self.config.context_size
            for dim in range(self.config.embedding_dim)
        ]

    def _add_context_projection_scalars(
        self,
        hidden: list[Scalar],
        x: list[list[Scalar]],
    ) -> list[Scalar]:
        context_summary = [
            sum(row[dim] for row in x) / self.config.context_size
            for dim in range(self.config.embedding_dim)
        ]
        projected_summary = linear_scalars(
            context_summary,
            self.context_projection_w,
            self.context_projection_b,
        )
        return [
            hidden[dim] + projected_summary[dim]
            for dim in range(self.config.embedding_dim)
        ]

    def _add_prompt_prefix_projection_scalars(
        self,
        hidden: list[Scalar],
        x: list[list[Scalar]],
        context: list[int],
        last_position: int,
    ) -> list[Scalar]:
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
        projected_summary = linear_scalars(
            prompt_summary,
            self.prompt_prefix_projection_w,
            self.prompt_prefix_projection_b,
        )
        return [
            hidden[dim] + projected_summary[dim]
            for dim in range(self.config.embedding_dim)
        ]

    def _add_prompt_position_projection_scalars(
        self,
        hidden: list[Scalar],
        x: list[list[Scalar]],
        context: list[int],
        last_position: int,
    ) -> list[Scalar]:
        prompt_positions = [
            (position, row)
            for position, row in enumerate(x[:last_position])
            if context[position] != 0
        ]
        if not prompt_positions:
            return hidden
        projected_summary: list[Scalar] = []
        for output_dim, bias in enumerate(self.prompt_position_projection_b):
            total = Scalar(0.0)
            for position, row in prompt_positions:
                position_weights = self.prompt_position_projection_w[position]
                for input_dim, value in enumerate(row):
                    total = total + value * position_weights[input_dim][output_dim]
            projected_summary.append(total / len(prompt_positions) + bias)
        return [
            hidden[dim]
            + projected_summary[dim] * self.config.prompt_position_projection_scale
            for dim in range(self.config.embedding_dim)
        ]

    def _add_prompt_attention_summary_scalars(
        self,
        hidden: list[Scalar],
        x: list[list[Scalar]],
    ) -> list[Scalar]:
        scores = [
            dot_scalars(self.prompt_summary_query, row)
            * (1.0 / math.sqrt(self.config.embedding_dim))
            for row in x
        ]
        weights = softmax_scalars(scores)
        attention_summary = []
        for dim in range(self.config.embedding_dim):
            total = Scalar(0.0)
            for row, weight in zip(x, weights):
                total = total + weight * row[dim]
            attention_summary.append(total)
        projected_summary = linear_scalars(
            attention_summary,
            self.prompt_summary_w,
            self.prompt_summary_b,
        )
        return [
            hidden[dim] + projected_summary[dim]
            for dim in range(self.config.embedding_dim)
        ]
