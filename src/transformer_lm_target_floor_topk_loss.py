"""Top-k target-floor pressure for routing-repair objectives."""

from __future__ import annotations

from collections import Counter
from typing import Any

from autograd import Scalar
from replay_plan import BranchReplayRecord, branch_replay_parts
from transformer_lm_branch_context_balancing import balanced_target_loss
from transformer_lm_branch_context_targets import branch_context_profile_key
from transformer_lm_candidate_set_loss import candidate_set_target_loss
from transformer_math import softmax_scalars


def target_floor_topk_loss(
    model: Any,
    target_floor_anchors: list[BranchReplayRecord],
    candidate_weight: float,
    candidate_count: int,
) -> Scalar | None:
    """Lift target-floor anchors against their current hard candidates."""

    if not target_floor_anchors or candidate_weight <= 0.0:
        return None
    loss = Scalar(0.0)
    count = 0
    for anchor in target_floor_anchors:
        context, target, _predicted, _profile = branch_replay_parts(anchor)
        logits = model._forward_scalars(context)
        loss = loss + candidate_set_target_loss(
            logits,
            target,
            candidate_weight,
            candidate_count,
            model.config.vocab_size,
        )
        count += 1
    if count == 0:
        return None
    return loss / count


def target_floor_combined_topk_loss(
    model: Any,
    target_floor_anchors: list[BranchReplayRecord],
    candidate_weight: float,
    candidate_count: int,
    competitor_weight: float = 0.0,
) -> Scalar | None:
    """Return balanced floor, top-k, and competitor pressure in one pass."""

    if (
        not target_floor_anchors
        or (candidate_weight <= 0.0 and competitor_weight <= 0.0)
    ):
        return None
    floor_losses_by_target: dict[tuple[str, int], Scalar] = {}
    floor_counts_by_target: Counter[tuple[str, int]] = Counter()
    candidate_loss = Scalar(0.0)
    competitor_loss = Scalar(0.0)
    anchor_count = 0
    for anchor in target_floor_anchors:
        context, target, _predicted, profile = branch_replay_parts(anchor)
        profile_key = branch_context_profile_key(profile, profile_aware_targets=True)
        target_key = (profile_key, target)
        logits = model._forward_scalars(context)
        probs = softmax_scalars(logits)
        floor_loss = -(probs[target] + 1e-12).log()
        floor_losses_by_target[target_key] = (
            floor_losses_by_target.get(target_key, Scalar(0.0)) + floor_loss
        )
        floor_counts_by_target[target_key] += 1
        candidate_loss = candidate_loss + candidate_set_target_loss(
            logits,
            target,
            candidate_weight,
            candidate_count,
            model.config.vocab_size,
        )
        competitor_loss = competitor_loss + _competitor_unlikelihood_loss(
            probs,
            target,
            competitor_weight,
        )
        anchor_count += 1
    if anchor_count == 0:
        return None
    balanced_floor_loss = (
        balanced_target_loss(floor_losses_by_target, floor_counts_by_target)
        * candidate_weight
    )
    return (
        balanced_floor_loss
        + (candidate_loss / anchor_count)
        + (competitor_loss / anchor_count)
    )


def _competitor_unlikelihood_loss(
    probs: list[Scalar],
    target: int,
    competitor_weight: float,
) -> Scalar:
    if competitor_weight <= 0.0:
        return Scalar(0.0)
    competitor = _top_competitor_id(probs, target)
    if competitor is None:
        return Scalar(0.0)
    return (-(Scalar(1.0) - probs[competitor] + 1e-12).log()) * competitor_weight


def _top_competitor_id(probs: list[Scalar], target: int) -> int | None:
    ranked_ids = sorted(
        range(len(probs)),
        key=lambda index: (-probs[index].data, index),
    )
    if not ranked_ids or ranked_ids[0] == target:
        return None
    return ranked_ids[0]
