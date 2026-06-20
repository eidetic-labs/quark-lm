"""Retention-floor loss shared by branch repair objectives."""

from __future__ import annotations

from typing import Any

from autograd import Scalar
from replay_plan import BranchReplayRecord
from transformer_lm_branch_context_floor_loss import (
    accumulate_floor_preservation_losses,
    floor_preservation_loss_contribution,
)
from transformer_lm_branch_context_loss_state import BranchContextLossState
from transformer_lm_branch_context_targets import (
    build_branch_context_replay_target_state,
)


def retention_floor_loss(
    model: Any,
    retention_anchors: list[BranchReplayRecord],
    retention_weight: float,
) -> Scalar | None:
    """Return preservation loss for already represented branch targets."""

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
