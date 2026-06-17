"""Training orchestration for branch-context replay objectives."""

from __future__ import annotations

from typing import Any

from autograd import Scalar, zero_grad
from replay_plan import BranchReplayRecord
from transformer_lm_branch_context_branch_loss import accumulate_branch_losses
from transformer_lm_branch_context_floor_loss import (
    accumulate_floor_preservation_losses,
    floor_preservation_loss_contribution,
)
from transformer_lm_branch_context_loss_state import BranchContextLossState
from transformer_lm_branch_context_replay_contribution import replay_loss_contribution
from transformer_lm_branch_context_replay_loss import accumulate_replay_losses
from transformer_lm_branch_context_targets import (
    build_branch_context_replay_target_state,
)


def train_branch_context_replay_coverage(
    model: Any,
    branches: list[BranchReplayRecord],
    replay_branches: list[BranchReplayRecord],
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    replay_weight: float,
    hard_negative_count: int,
    params: list[Scalar] | None = None,
    preserve_covered_targets: bool = False,
    balance_covered_target_anchors: bool = False,
    focus_uncovered_targets: bool = False,
    preserve_predicted_target_coverage: bool = False,
    balance_deficit_targets: bool = False,
    profile_aware_targets: bool = False,
    balance_profile_target_shares: bool = False,
    enforce_prompt_target_margins: bool = False,
    floor_preservation_branches: list[BranchReplayRecord] | None = None,
    floor_preservation_weight: float = 0.0,
    balance_floor_preservation_targets: bool = False,
) -> float:
    params = model.parameters() if params is None else params
    zero_grad(params)
    target_state = build_branch_context_replay_target_state(
        branches,
        replay_branches,
        floor_preservation_branches,
        profile_aware_targets=profile_aware_targets,
    )
    losses = BranchContextLossState()
    accumulate_branch_losses(
        model,
        target_state.branch_parts,
        losses,
        positive_weight=positive_weight,
        negative_weight=negative_weight,
    )
    accumulate_replay_losses(
        model,
        target_state,
        losses,
        hard_negative_count=hard_negative_count,
        focus_uncovered_targets=focus_uncovered_targets,
        preserve_predicted_target_coverage=preserve_predicted_target_coverage,
        preserve_covered_targets=preserve_covered_targets,
        profile_aware_targets=profile_aware_targets,
        balance_profile_target_shares=balance_profile_target_shares,
        enforce_prompt_target_margins=enforce_prompt_target_margins,
    )
    accumulate_floor_preservation_losses(
        model,
        target_state,
        losses,
        profile_aware_targets=profile_aware_targets,
        floor_preservation_weight=floor_preservation_weight,
    )

    loss = losses.branch_loss / max(len(branches), 1)
    if replay_weight > 0.0:
        replay_loss = replay_loss_contribution(
            losses,
            target_state,
            focus_uncovered_targets=focus_uncovered_targets,
            balance_deficit_targets=balance_deficit_targets,
            balance_profile_target_shares=balance_profile_target_shares,
            enforce_prompt_target_margins=enforce_prompt_target_margins,
            preserve_predicted_target_coverage=preserve_predicted_target_coverage,
            preserve_covered_targets=preserve_covered_targets,
            balance_covered_target_anchors=balance_covered_target_anchors,
        )
        if replay_loss is not None:
            loss = loss + replay_loss * replay_weight
    floor_loss = floor_preservation_loss_contribution(
        losses,
        floor_preservation_weight=floor_preservation_weight,
        balance_floor_preservation_targets=balance_floor_preservation_targets,
    )
    if floor_loss is not None:
        loss = loss + floor_loss
    loss.backward()
    model.apply_gradients(params, learning_rate)
    return loss.data
