"""Retention-anchored rank objective mixin for TinyTransformerLM."""

from __future__ import annotations

from autograd import Scalar
from replay_plan import BranchReplayRecord
from transformer_lm_retention_rank_training import (
    RankBranchRecord,
    train_branch_retention_rank_margin,
)


class TransformerRetentionRankObjectiveMixin:
    def train_step_with_branch_retention_rank_margin(
        self,
        branches: list[RankBranchRecord],
        retention_anchors: list[BranchReplayRecord],
        learning_rate: float,
        negative_weight: float,
        positive_weight: float,
        margin_weight: float,
        hard_negative_count: int,
        params: list[Scalar] | None = None,
    ) -> float:
        return train_branch_retention_rank_margin(
            self,
            branches,
            retention_anchors,
            learning_rate,
            negative_weight,
            positive_weight,
            margin_weight,
            hard_negative_count,
            params=params,
        )
