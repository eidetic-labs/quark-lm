"""Assembled tiny decoder-only transformer language model."""

from __future__ import annotations

from typing import Any

from transformer_lm_blocks import TransformerBlockMixin
from transformer_lm_branch_binding import TransformerBranchBindingMixin
from transformer_lm_branch_context import TransformerBranchContextMixin
from transformer_lm_branch_target_objectives import (
    TransformerBranchTargetObjectiveMixin,
)
from transformer_lm_forward import TransformerForwardMixin
from transformer_lm_generation import TransformerGenerationIOMixin
from transformer_lm_objectives import TransformerObjectiveMixin
from transformer_lm_rank_objectives import TransformerRankObjectiveMixin
from transformer_lm_rank_collapse_objectives import (
    TransformerRankCollapseObjectiveMixin,
)
from transformer_lm_retention_rank_objectives import (
    TransformerRetentionRankObjectiveMixin,
)
from transformer_lm_retention_topk_objectives import (
    TransformerRetentionTopKObjectiveMixin,
)
from transformer_lm_weights import TransformerWeightsMixin
from transformer_math import matrix_to_scalars, vector_to_scalars
from transformer_model import TransformerConfig, validate_transformer_config
from transformer_optimizer import ScalarOptimizer


class TinyTransformerLM(
    TransformerWeightsMixin,
    TransformerForwardMixin,
    TransformerBlockMixin,
    TransformerObjectiveMixin,
    TransformerBranchTargetObjectiveMixin,
    TransformerBranchBindingMixin,
    TransformerBranchContextMixin,
    TransformerRankObjectiveMixin,
    TransformerRankCollapseObjectiveMixin,
    TransformerRetentionRankObjectiveMixin,
    TransformerRetentionTopKObjectiveMixin,
    TransformerGenerationIOMixin,
):
    def __init__(self, config: TransformerConfig, weights: dict[str, Any]) -> None:
        validate_transformer_config(config)
        self.config = config
        dim = config.embedding_dim
        self.token_embeddings = matrix_to_scalars(weights["token_embeddings"])
        self.position_embeddings = matrix_to_scalars(weights["position_embeddings"])
        self.wq = matrix_to_scalars(weights["wq"])
        self.bq = vector_to_scalars(weights["bq"])
        self.wk = matrix_to_scalars(weights["wk"])
        self.bk = vector_to_scalars(weights["bk"])
        self.wv = matrix_to_scalars(weights["wv"])
        self.bv = vector_to_scalars(weights["bv"])
        self.wo = matrix_to_scalars(weights["wo"])
        self.bo = vector_to_scalars(weights["bo"])
        self.w1 = matrix_to_scalars(weights["w1"])
        self.b1 = vector_to_scalars(weights["b1"])
        self.w_gate = matrix_to_scalars(
            weights.get(
                "w_gate",
                [[0.0 for _ in range(config.feedforward_dim)] for _ in range(dim)],
            )
        )
        self.b_gate = vector_to_scalars(
            weights.get("b_gate", [0.0 for _ in range(config.feedforward_dim)])
        )
        self.w2 = matrix_to_scalars(weights["w2"])
        self.b2 = vector_to_scalars(weights["b2"])
        self.wout = matrix_to_scalars(weights["wout"])
        self.bout = vector_to_scalars(weights["bout"])
        self.context_projection_w = matrix_to_scalars(
            weights.get(
                "context_projection_w",
                [[0.0 for _ in range(dim)] for _ in range(dim)],
            )
        )
        self.context_projection_b = vector_to_scalars(
            weights.get("context_projection_b", [0.0 for _ in range(dim)])
        )
        self.prompt_prefix_projection_w = matrix_to_scalars(
            weights.get(
                "prompt_prefix_projection_w",
                [[0.0 for _ in range(dim)] for _ in range(dim)],
            )
        )
        self.prompt_prefix_projection_b = vector_to_scalars(
            weights.get("prompt_prefix_projection_b", [0.0 for _ in range(dim)])
        )
        self.prompt_position_projection_w = [
            matrix_to_scalars(position_weights)
            for position_weights in weights.get(
                "prompt_position_projection_w",
                [
                    [[0.0 for _ in range(dim)] for _ in range(dim)]
                    for _ in range(config.context_size)
                ],
            )
        ]
        self.prompt_position_projection_b = vector_to_scalars(
            weights.get("prompt_position_projection_b", [0.0 for _ in range(dim)])
        )
        self.prompt_summary_query = vector_to_scalars(
            weights.get("prompt_summary_query", [0.0 for _ in range(dim)])
        )
        self.prompt_summary_w = matrix_to_scalars(
            weights.get(
                "prompt_summary_w",
                [[0.0 for _ in range(dim)] for _ in range(dim)],
            )
        )
        self.prompt_summary_b = vector_to_scalars(
            weights.get("prompt_summary_b", [0.0 for _ in range(dim)])
        )
        self.ln1_gain = vector_to_scalars(
            weights.get("ln1_gain", [1.0 for _ in range(dim)])
        )
        self.ln1_bias = vector_to_scalars(
            weights.get("ln1_bias", [0.0 for _ in range(dim)])
        )
        self.ln2_gain = vector_to_scalars(
            weights.get("ln2_gain", [1.0 for _ in range(dim)])
        )
        self.ln2_bias = vector_to_scalars(
            weights.get("ln2_bias", [0.0 for _ in range(dim)])
        )
        self.final_ln_gain = vector_to_scalars(
            weights.get("final_ln_gain", [1.0 for _ in range(dim)])
        )
        self.final_ln_bias = vector_to_scalars(
            weights.get("final_ln_bias", [0.0 for _ in range(dim)])
        )
        self.extra_blocks = [
            self._block_from_dict(layer)
            for layer in weights.get("extra_layers", [])
        ]
        expected_extra_layers = max(config.num_layers - 1, 0)
        if len(self.extra_blocks) != expected_extra_layers:
            raise ValueError(
                f"checkpoint has {len(self.extra_blocks)} extra transformer layers, "
                f"config expects {expected_extra_layers}"
            )
        self.blocks = [self._first_block()] + self.extra_blocks
        self.freeze_lower_layers_for_updates = False
        self.active_optimizer: ScalarOptimizer | None = None
