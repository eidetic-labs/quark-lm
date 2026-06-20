"""Single-pass retention-anchored top-k softmax training."""

from __future__ import annotations

from typing import Any

from autograd import Scalar, zero_grad
from replay_plan import BranchReplayRecord
from transformer_lm_branch_retention_floor_loss import retention_floor_loss
from transformer_lm_candidate_set_loss import candidate_set_target_loss
from transformer_lm_target_floor_representation_loss import (
    target_floor_representation_loss,
)
from transformer_lm_target_floor_topk_loss import target_floor_combined_topk_loss
from transformer_math import softmax_scalars

TopKBranchRecord = tuple[list[int], int, int]


def train_branch_retention_topk_softmax(
    model: Any,
    branches: list[TopKBranchRecord],
    retention_anchors: list[BranchReplayRecord],
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    candidate_weight: float,
    candidate_count: int,
    params: list[Scalar] | None = None,
    target_floor_anchors: list[BranchReplayRecord] | None = None,
    representation_weight: float = 0.0,
) -> float:
    """Apply top-k pressure and retention preservation in one optimizer step."""

    params = model.parameters() if params is None else params
    zero_grad(params)
    loss = _topk_softmax_loss(
        model,
        branches,
        negative_weight,
        positive_weight,
        candidate_weight,
        candidate_count,
    )
    anchor_loss = retention_floor_loss(model, retention_anchors, candidate_weight)
    if anchor_loss is not None:
        loss = loss + anchor_loss
    target_floor_loss = target_floor_combined_topk_loss(
        model,
        target_floor_anchors or [],
        candidate_weight,
        candidate_count,
        negative_weight,
    )
    if target_floor_loss is not None:
        loss = loss + target_floor_loss
    representation_loss = target_floor_representation_loss(
        model,
        branches,
        target_floor_anchors or [],
        representation_weight,
    )
    if representation_loss is not None:
        loss = loss + representation_loss
    loss.backward()
    model.apply_gradients(params, learning_rate)
    return loss.data


def _topk_softmax_loss(
    model: Any,
    branches: list[TopKBranchRecord],
    negative_weight: float,
    positive_weight: float,
    candidate_weight: float,
    candidate_count: int,
) -> Scalar:
    loss = Scalar(0.0)
    for context, target, predicted in branches:
        logits = model._forward_scalars(context)
        probs = softmax_scalars(logits)
        if positive_weight > 0.0:
            loss = loss + (-probs[target].log()) * positive_weight
        if negative_weight > 0.0 and predicted != target:
            loss = loss + (
                -(Scalar(1.0) - probs[predicted] + 1e-12).log()
            ) * negative_weight
        loss = loss + candidate_set_target_loss(
            logits,
            target,
            candidate_weight,
            candidate_count,
            model.config.vocab_size,
        )
    return loss / max(len(branches), 1)
