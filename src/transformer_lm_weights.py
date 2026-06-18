"""Weight initialization and parameter views for TinyTransformerLM."""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Any

from autograd import Scalar
from transformer_math import (
    matrix_to_floats,
    matrix_to_scalars,
    vector_to_floats,
    vector_to_scalars,
)
from transformer_lm_parameter_views import (
    all_transformer_parameters,
    top_layer_transformer_parameters,
    uses_block_layer_norm_parameters,
)
from transformer_lm_weight_initialization import build_random_transformer_weights
from transformer_model import TransformerConfig


class TransformerWeightsMixin:
    @classmethod
    def init_random(cls, config: TransformerConfig) -> "TinyTransformerLM":
        return cls(config, build_random_transformer_weights(config))

    def _first_block(self) -> dict[str, Any]:
        return {
            "wq": self.wq,
            "bq": self.bq,
            "wk": self.wk,
            "bk": self.bk,
            "wv": self.wv,
            "bv": self.bv,
            "wo": self.wo,
            "bo": self.bo,
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
            "w_gate": self.w_gate,
            "b_gate": self.b_gate,
            "ln1_gain": self.ln1_gain,
            "ln1_bias": self.ln1_bias,
            "ln2_gain": self.ln2_gain,
            "ln2_bias": self.ln2_bias,
        }

    def _block_from_dict(self, payload: dict[str, Any]) -> dict[str, Any]:
        dim = self.config.embedding_dim
        return {
            "wq": matrix_to_scalars(payload["wq"]),
            "bq": vector_to_scalars(payload["bq"]),
            "wk": matrix_to_scalars(payload["wk"]),
            "bk": vector_to_scalars(payload["bk"]),
            "wv": matrix_to_scalars(payload["wv"]),
            "bv": vector_to_scalars(payload["bv"]),
            "wo": matrix_to_scalars(payload["wo"]),
            "bo": vector_to_scalars(payload["bo"]),
            "w1": matrix_to_scalars(payload["w1"]),
            "b1": vector_to_scalars(payload["b1"]),
            "w_gate": matrix_to_scalars(
                payload.get(
                    "w_gate",
                    [[0.0 for _ in range(self.config.feedforward_dim)] for _ in range(dim)],
                )
            ),
            "b_gate": vector_to_scalars(
                payload.get("b_gate", [0.0 for _ in range(self.config.feedforward_dim)])
            ),
            "w2": matrix_to_scalars(payload["w2"]),
            "b2": vector_to_scalars(payload["b2"]),
            "ln1_gain": vector_to_scalars(payload.get("ln1_gain", [1.0 for _ in range(dim)])),
            "ln1_bias": vector_to_scalars(payload.get("ln1_bias", [0.0 for _ in range(dim)])),
            "ln2_gain": vector_to_scalars(payload.get("ln2_gain", [1.0 for _ in range(dim)])),
            "ln2_bias": vector_to_scalars(payload.get("ln2_bias", [0.0 for _ in range(dim)])),
        }

    def _block_to_floats(self, block: dict[str, Any]) -> dict[str, Any]:
        return {
            "wq": matrix_to_floats(block["wq"]),
            "bq": vector_to_floats(block["bq"]),
            "wk": matrix_to_floats(block["wk"]),
            "bk": vector_to_floats(block["bk"]),
            "wv": matrix_to_floats(block["wv"]),
            "bv": vector_to_floats(block["bv"]),
            "wo": matrix_to_floats(block["wo"]),
            "bo": vector_to_floats(block["bo"]),
            "w1": matrix_to_floats(block["w1"]),
            "b1": vector_to_floats(block["b1"]),
            "w_gate": matrix_to_floats(block["w_gate"]),
            "b_gate": vector_to_floats(block["b_gate"]),
            "w2": matrix_to_floats(block["w2"]),
            "b2": vector_to_floats(block["b2"]),
            "ln1_gain": vector_to_floats(block["ln1_gain"]),
            "ln1_bias": vector_to_floats(block["ln1_bias"]),
            "ln2_gain": vector_to_floats(block["ln2_gain"]),
            "ln2_bias": vector_to_floats(block["ln2_bias"]),
        }

    def _uses_block_layer_norm_parameters(self) -> bool:
        return uses_block_layer_norm_parameters(self.config)

    def parameters(self) -> list[Scalar]:
        return all_transformer_parameters(self)

    def top_layer_parameters(self) -> list[Scalar]:
        return top_layer_transformer_parameters(self)

    def resize_vocab(self, vocab_size: int) -> None:
        if vocab_size < self.config.vocab_size:
            raise ValueError("vocab_size cannot shrink")
        if vocab_size == self.config.vocab_size:
            return

        old_vocab_size = self.config.vocab_size
        dim = self.config.embedding_dim
        for token_id in range(old_vocab_size, vocab_size):
            self.token_embeddings.append(
                [
                    Scalar(self._new_vocab_weight(token_id, index, salt=1))
                    for index in range(dim)
                ]
            )
            self.bout.append(Scalar(0.0))
        for hidden_dim, row in enumerate(self.wout):
            for token_id in range(old_vocab_size, vocab_size):
                row.append(
                    Scalar(self._new_vocab_weight(token_id, hidden_dim, salt=2))
                )
        self.config = replace(self.config, vocab_size=vocab_size)

    def _new_vocab_weight(self, token_id: int, index: int, *, salt: int) -> float:
        rng = random.Random(self.config.seed + token_id * 1009 + index * 37 + salt)
        return rng.uniform(-0.05, 0.05)

    def _output_weights_scalars(self) -> list[list[Scalar]]:
        if not self.config.tie_output_embeddings:
            return self.wout
        return [
            [self.token_embeddings[token_id][dim] for token_id in range(self.config.vocab_size)]
            for dim in range(self.config.embedding_dim)
        ]

    def _output_weights_floats(self) -> list[list[float]]:
        if not self.config.tie_output_embeddings:
            return matrix_to_floats(self.wout)
        token_embeddings = matrix_to_floats(self.token_embeddings)
        return [
            [token_embeddings[token_id][dim] for token_id in range(self.config.vocab_size)]
            for dim in range(self.config.embedding_dim)
        ]
