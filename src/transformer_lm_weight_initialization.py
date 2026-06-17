"""Random weight construction for TinyTransformerLM."""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from typing import Any

from transformer_model import TransformerConfig, validate_transformer_config


def build_random_transformer_weights(config: TransformerConfig) -> dict[str, Any]:
    validate_transformer_config(config)
    rng = random.Random(config.seed)

    def rand(scale: float) -> float:
        return rng.uniform(-scale, scale)

    dim = config.embedding_dim
    scale = 1.0 / math.sqrt(dim)
    first_block = build_random_transformer_block(config, rand)
    return {
        "token_embeddings": [
            [rand(0.08) for _ in range(dim)]
            for _ in range(config.vocab_size)
        ],
        "position_embeddings": [
            [rand(0.08) for _ in range(dim)]
            for _ in range(config.context_size)
        ],
        **first_block,
        "wout": [[rand(scale) for _ in range(config.vocab_size)] for _ in range(dim)],
        "bout": [0.0 for _ in range(config.vocab_size)],
        "context_projection_w": [[0.0 for _ in range(dim)] for _ in range(dim)],
        "context_projection_b": [0.0 for _ in range(dim)],
        "prompt_prefix_projection_w": [[0.0 for _ in range(dim)] for _ in range(dim)],
        "prompt_prefix_projection_b": [0.0 for _ in range(dim)],
        "prompt_position_projection_w": [
            [[0.0 for _ in range(dim)] for _ in range(dim)]
            for _ in range(config.context_size)
        ],
        "prompt_position_projection_b": [0.0 for _ in range(dim)],
        "prompt_summary_query": [rand(scale) for _ in range(dim)],
        "prompt_summary_w": [[0.0 for _ in range(dim)] for _ in range(dim)],
        "prompt_summary_b": [0.0 for _ in range(dim)],
        "final_ln_gain": [1.0 for _ in range(dim)],
        "final_ln_bias": [0.0 for _ in range(dim)],
        "extra_layers": [
            build_random_transformer_block(config, rand)
            for _ in range(max(config.num_layers - 1, 0))
        ],
    }


def build_random_transformer_block(
    config: TransformerConfig,
    rand: Callable[[float], float],
) -> dict[str, Any]:
    dim = config.embedding_dim
    ff_dim = config.feedforward_dim
    scale = 1.0 / math.sqrt(dim)
    return {
        "wq": [[rand(scale) for _ in range(dim)] for _ in range(dim)],
        "bq": [0.0 for _ in range(dim)],
        "wk": [[rand(scale) for _ in range(dim)] for _ in range(dim)],
        "bk": [0.0 for _ in range(dim)],
        "wv": [[rand(scale) for _ in range(dim)] for _ in range(dim)],
        "bv": [0.0 for _ in range(dim)],
        "wo": [[rand(scale) for _ in range(dim)] for _ in range(dim)],
        "bo": [0.0 for _ in range(dim)],
        "w1": [[rand(scale) for _ in range(ff_dim)] for _ in range(dim)],
        "b1": [0.0 for _ in range(ff_dim)],
        "w_gate": [
            [rand(scale) if config.use_gated_mlp else 0.0 for _ in range(ff_dim)]
            for _ in range(dim)
        ],
        "b_gate": [0.0 for _ in range(ff_dim)],
        "w2": [
            [rand(1.0 / math.sqrt(ff_dim)) for _ in range(dim)]
            for _ in range(ff_dim)
        ],
        "b2": [0.0 for _ in range(dim)],
        "ln1_gain": [1.0 for _ in range(dim)],
        "ln1_bias": [0.0 for _ in range(dim)],
        "ln2_gain": [1.0 for _ in range(dim)],
        "ln2_bias": [0.0 for _ in range(dim)],
    }
