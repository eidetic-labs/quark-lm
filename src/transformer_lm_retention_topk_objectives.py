"""Retention-anchored top-k objective mixin for TinyTransformerLM."""

from __future__ import annotations

from autograd import Scalar
from replay_plan import BranchReplayRecord
from transformer_lm_retention_topk_training import (
    TopKBranchRecord,
    train_branch_retention_topk_softmax,
)


class TransformerRetentionTopKObjectiveMixin:
    def train_step_with_branch_retention_topk_softmax(
        self,
        branches: list[TopKBranchRecord],
        retention_anchors: list[BranchReplayRecord],
        learning_rate: float,
        negative_weight: float,
        positive_weight: float,
        candidate_weight: float,
        candidate_count: int,
        params: list[Scalar] | None = None,
        target_floor_anchors: list[BranchReplayRecord] | None = None,
    ) -> float:
        return train_branch_retention_topk_softmax(
            self,
            branches,
            retention_anchors,
            learning_rate,
            negative_weight,
            positive_weight,
            candidate_weight,
            candidate_count,
            params=params,
            target_floor_anchors=target_floor_anchors,
        )
