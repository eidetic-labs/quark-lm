"""Replay-loss reduction for branch-context objectives."""

from __future__ import annotations

from autograd import Scalar
from transformer_lm_branch_context_balancing import balanced_target_loss
from transformer_lm_branch_context_loss_state import BranchContextLossState
from transformer_lm_branch_context_targets import BranchContextReplayTargetState


def replay_loss_contribution(
    losses: BranchContextLossState,
    target_state: BranchContextReplayTargetState,
    *,
    focus_uncovered_targets: bool,
    balance_deficit_targets: bool,
    balance_profile_target_shares: bool,
    enforce_prompt_target_margins: bool,
    preserve_predicted_target_coverage: bool,
    preserve_covered_targets: bool,
    balance_covered_target_anchors: bool,
) -> Scalar | None:
    if not (target_state.replay_record_count and target_state.replay_targets):
        return None
    replay_loss = (
        losses.replay_coverage_loss / target_state.replay_record_count
        + losses.replay_ownership_loss / target_state.replay_record_count
    ) / 2.0
    if focus_uncovered_targets and losses.deficit_target_count:
        replay_loss = replay_loss + deficit_loss_contribution(
            losses,
            balance_deficit_targets=balance_deficit_targets,
        )
    if balance_profile_target_shares and losses.profile_target_share_losses_by_target:
        replay_loss = replay_loss + balanced_target_loss(
            losses.profile_target_share_losses_by_target,
            losses.profile_target_share_counts_by_target,
        )
    if enforce_prompt_target_margins and losses.prompt_target_margin_count:
        replay_loss = replay_loss + (
            losses.prompt_target_margin_loss / losses.prompt_target_margin_count
        )
    if (
        preserve_predicted_target_coverage
        and losses.coverage_preservation_losses_by_target
    ):
        replay_loss = replay_loss + balanced_target_loss(
            losses.coverage_preservation_losses_by_target,
            losses.coverage_preservation_counts_by_target,
        )
    if (
        preserve_covered_targets
        and balance_covered_target_anchors
        and len(losses.covered_anchor_losses_by_target) > 1
    ):
        replay_loss = replay_loss + balanced_target_loss(
            losses.covered_anchor_losses_by_target,
            losses.covered_anchor_counts_by_target,
        )
    elif (
        preserve_covered_targets
        and not balance_covered_target_anchors
        and losses.covered_anchor_count
    ):
        replay_loss = replay_loss + (
            losses.covered_anchor_loss / max(losses.covered_anchor_count, 1)
        )
    return replay_loss


def deficit_loss_contribution(
    losses: BranchContextLossState,
    *,
    balance_deficit_targets: bool,
) -> Scalar:
    if balance_deficit_targets and losses.deficit_target_losses_by_target:
        return balanced_target_loss(
            losses.deficit_target_losses_by_target,
            losses.deficit_target_counts_by_target,
        )
    return losses.deficit_target_loss / max(losses.deficit_target_count, 1)
