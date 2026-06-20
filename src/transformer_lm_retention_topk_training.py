"""Single-pass retention-anchored top-k softmax training."""

from __future__ import annotations

from typing import Any

from autograd import Scalar, zero_grad
from replay_plan import BranchReplayRecord
from transformer_lm_branch_retention_floor_loss import retention_floor_loss
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
        loss = loss + _candidate_loss(
            logits,
            target,
            candidate_weight,
            candidate_count,
            model.config.vocab_size,
        )
    return loss / max(len(branches), 1)


def _candidate_loss(
    logits: list[Scalar],
    target: int,
    candidate_weight: float,
    candidate_count: int,
    vocab_size: int,
) -> Scalar:
    if candidate_weight <= 0.0:
        return Scalar(0.0)
    hard_candidates = [
        index
        for index in sorted(
            range(vocab_size),
            key=lambda item: logits[item].data,
            reverse=True,
        )
        if index != target
    ]
    if candidate_count > 0:
        hard_candidates = hard_candidates[:candidate_count]
    candidate_ids = [target, *hard_candidates]
    if len(candidate_ids) <= 1:
        return Scalar(0.0)
    candidate_logits = [logits[candidate_id] for candidate_id in candidate_ids]
    candidate_probs = softmax_scalars(candidate_logits)
    return (-candidate_probs[0].log()) * candidate_weight
