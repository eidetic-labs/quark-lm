"""Floor-preservation losses for branch-context objectives."""

from __future__ import annotations

from typing import Any

from autograd import Scalar
from transformer_lm_branch_context_balancing import balanced_target_loss
from transformer_lm_branch_context_loss_state import BranchContextLossState
from transformer_lm_branch_context_targets import (
    BranchContextReplayTargetState,
    branch_context_profile_key,
)
from transformer_math import softmax_scalars


def accumulate_floor_preservation_losses(
    model: Any,
    target_state: BranchContextReplayTargetState,
    losses: BranchContextLossState,
    *,
    profile_aware_targets: bool,
    floor_preservation_weight: float,
) -> None:
    if floor_preservation_weight <= 0.0:
        return
    for context, target, _predicted, profile in target_state.floor_preservation_parts:
        profile_key = branch_context_profile_key(profile, profile_aware_targets)
        target_key = (profile_key, target)
        probs = softmax_scalars(model._forward_scalars(context))
        target_floor_loss = -(probs[target] + 1e-12).log()
        losses.floor_preservation_loss = (
            losses.floor_preservation_loss + target_floor_loss
        )
        losses.floor_preservation_count += 1
        losses.floor_preservation_losses_by_target[target_key] = (
            losses.floor_preservation_losses_by_target.get(target_key, Scalar(0.0))
            + target_floor_loss
        )
        losses.floor_preservation_counts_by_target[target_key] += 1


def floor_preservation_loss_contribution(
    losses: BranchContextLossState,
    *,
    floor_preservation_weight: float,
    balance_floor_preservation_targets: bool,
) -> Scalar | None:
    if floor_preservation_weight <= 0.0 or not losses.floor_preservation_count:
        return None
    if (
        balance_floor_preservation_targets
        and losses.floor_preservation_losses_by_target
    ):
        floor_loss = balanced_target_loss(
            losses.floor_preservation_losses_by_target,
            losses.floor_preservation_counts_by_target,
        )
    else:
        floor_loss = losses.floor_preservation_loss / max(
            losses.floor_preservation_count,
            1,
        )
    return floor_loss * floor_preservation_weight
