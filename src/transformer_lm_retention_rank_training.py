"""Single-pass retention-anchored rank-margin training."""

from __future__ import annotations

from typing import Any

from autograd import Scalar, zero_grad
from replay_plan import BranchReplayRecord
from transformer_lm_branch_context_floor_loss import (
    accumulate_floor_preservation_losses,
    floor_preservation_loss_contribution,
)
from transformer_lm_branch_context_loss_state import BranchContextLossState
from transformer_lm_branch_context_targets import (
    build_branch_context_replay_target_state,
)
from transformer_math import softmax_scalars


RankBranchRecord = tuple[list[int], int, int]


def train_branch_retention_rank_margin(
    model: Any,
    branches: list[RankBranchRecord],
    retention_anchors: list[BranchReplayRecord],
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    margin_weight: float,
    hard_negative_count: int,
    params: list[Scalar] | None = None,
) -> float:
    """Apply rank pressure and retention preservation in one optimizer step."""

    params = model.parameters() if params is None else params
    zero_grad(params)
    loss = _rank_margin_loss(
        model,
        branches,
        negative_weight,
        positive_weight,
        margin_weight,
        hard_negative_count,
    )
    retention_loss = _retention_floor_loss(
        model,
        retention_anchors,
        margin_weight,
    )
    if retention_loss is not None:
        loss = loss + retention_loss
    loss.backward()
    model.apply_gradients(params, learning_rate)
    return loss.data


def _rank_margin_loss(
    model: Any,
    branches: list[RankBranchRecord],
    negative_weight: float,
    positive_weight: float,
    margin_weight: float,
    hard_negative_count: int,
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
        loss = loss + _hard_negative_margin_loss(
            logits,
            target,
            margin_weight,
            hard_negative_count,
            model.config.vocab_size,
        )
    return loss / max(len(branches), 1)


def _hard_negative_margin_loss(
    logits: list[Scalar],
    target: int,
    margin_weight: float,
    hard_negative_count: int,
    vocab_size: int,
) -> Scalar:
    if margin_weight <= 0.0:
        return Scalar(0.0)
    hard_negatives = [
        index
        for index in sorted(
            range(vocab_size),
            key=lambda item: logits[item].data,
            reverse=True,
        )
        if index != target
    ]
    if hard_negative_count > 0:
        hard_negatives = hard_negatives[:hard_negative_count]
    if not hard_negatives:
        return Scalar(0.0)
    target_logit = logits[target]
    per_negative_weight = margin_weight / len(hard_negatives)
    loss = Scalar(0.0)
    for hard_negative in hard_negatives:
        gap = logits[hard_negative] - target_logit + 1.0
        loss = loss + (Scalar(1.0) + gap.exp()).log() * per_negative_weight
    return loss


def _retention_floor_loss(
    model: Any,
    retention_anchors: list[BranchReplayRecord],
    retention_weight: float,
) -> Scalar | None:
    if not retention_anchors or retention_weight <= 0.0:
        return None
    target_state = build_branch_context_replay_target_state(
        retention_anchors,
        retention_anchors,
        retention_anchors,
        profile_aware_targets=True,
    )
    losses = BranchContextLossState()
    accumulate_floor_preservation_losses(
        model,
        target_state,
        losses,
        profile_aware_targets=True,
        floor_preservation_weight=retention_weight,
    )
    return floor_preservation_loss_contribution(
        losses,
        floor_preservation_weight=retention_weight,
        balance_floor_preservation_targets=True,
    )
