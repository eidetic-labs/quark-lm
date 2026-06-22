"""Causal attention and rotary-position helpers for TinyTransformerLM."""

from __future__ import annotations

import math

from autograd import Scalar
from transformer_math import (
    dot_floats,
    dot_scalars,
    softmax_floats,
    softmax_scalars,
)


class TransformerAttentionMixin:
    def _causal_attention_scalars(
        self,
        q: list[list[Scalar]],
        k: list[list[Scalar]],
        v: list[list[Scalar]],
        position: int,
    ) -> list[Scalar]:
        head_dim = self.config.embedding_dim // self.config.attention_heads
        attended: list[Scalar] = []
        for head in range(self.config.attention_heads):
            start = head * head_dim
            end = start + head_dim
            scale = 1.0 / math.sqrt(head_dim)
            scores = [
                dot_scalars(q[position][start:end], k[past][start:end]) * scale
                for past in range(position + 1)
            ]
            weights = softmax_scalars(scores)
            for dim in range(start, end):
                total = Scalar(0.0)
                for past, weight in enumerate(weights):
                    total = total + weight * v[past][dim]
                attended.append(total)
        return attended

    def _causal_attention_floats(
        self,
        q: list[list[float]],
        k: list[list[float]],
        v: list[list[float]],
        position: int,
    ) -> list[float]:
        head_dim = self.config.embedding_dim // self.config.attention_heads
        attended: list[float] = []
        for head in range(self.config.attention_heads):
            start = head * head_dim
            end = start + head_dim
            scale = 1.0 / math.sqrt(head_dim)
            scores = [
                dot_floats(q[position][start:end], k[past][start:end]) * scale
                for past in range(position + 1)
            ]
            weights = softmax_floats(scores)
            for dim in range(start, end):
                attended.append(
                    sum(weight * v[past][dim] for past, weight in enumerate(weights))
                )
        return attended

    def _apply_rotary_scalars(
        self,
        rows: list[list[Scalar]],
        positions: list[int] | None = None,
    ) -> list[list[Scalar]]:
        # positions=None -> slot-keyed (enumerate), byte-identical to the pre-absolute
        # path. With absolute positions, a left-pad slot carries POSITION_PAD_SENTINEL
        # (< 0) and rotates by the IDENTITY (hard-coded cos=1.0/sin=0.0, not trig) so
        # the unmasked pad row passes through bit-exactly on f32/MPS.
        head_dim = self.config.embedding_dim // self.config.attention_heads
        rotated: list[list[Scalar]] = []
        for i, row in enumerate(rows):
            position = i if positions is None else positions[i]
            output = row[:]
            for head in range(self.config.attention_heads):
                start = head * head_dim
                for offset in range(0, head_dim - 1, 2):
                    index = start + offset
                    if position < 0:
                        cos_value, sin_value = 1.0, 0.0
                    else:
                        angle = position / (10000.0 ** (offset / max(head_dim, 1)))
                        cos_value = math.cos(angle)
                        sin_value = math.sin(angle)
                    left = row[index]
                    right = row[index + 1]
                    output[index] = left * cos_value - right * sin_value
                    output[index + 1] = left * sin_value + right * cos_value
            rotated.append(output)
        return rotated

    def _apply_rotary_floats(
        self,
        rows: list[list[float]],
        positions: list[int] | None = None,
    ) -> list[list[float]]:
        head_dim = self.config.embedding_dim // self.config.attention_heads
        rotated: list[list[float]] = []
        for i, row in enumerate(rows):
            position = i if positions is None else positions[i]
            output = row[:]
            for head in range(self.config.attention_heads):
                start = head * head_dim
                for offset in range(0, head_dim - 1, 2):
                    index = start + offset
                    if position < 0:
                        cos_value, sin_value = 1.0, 0.0
                    else:
                        angle = position / (10000.0 ** (offset / max(head_dim, 1)))
                        cos_value = math.cos(angle)
                        sin_value = math.sin(angle)
                    left = row[index]
                    right = row[index + 1]
                    output[index] = left * cos_value - right * sin_value
                    output[index + 1] = left * sin_value + right * cos_value
            rotated.append(output)
        return rotated
