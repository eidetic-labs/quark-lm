"""Optional replay-loss accumulators for branch-context objectives."""

from __future__ import annotations

from autograd import Scalar
from transformer_lm_branch_context_loss_state import BranchContextLossState
from transformer_lm_branch_context_targets import BranchContextReplayTargetState


def add_profile_target_share_loss(
    losses: BranchContextLossState,
    profile_key: str,
    target: int,
    owned_target_share: Scalar,
    *,
    active: bool,
) -> None:
    if not active:
        return
    target_key = (profile_key, target)
    target_share_loss = -(owned_target_share + 1e-12).log()
    losses.profile_target_share_losses_by_target[target_key] = (
        losses.profile_target_share_losses_by_target.get(target_key, Scalar(0.0))
        + target_share_loss
    )
    losses.profile_target_share_counts_by_target[target_key] += 1


def add_prompt_target_margin_loss(
    losses: BranchContextLossState,
    target: int,
    profile_targets: list[int],
    profile_offsets: dict[int, int],
    candidate_logits: list[Scalar],
    target_offset: int,
    *,
    active: bool,
) -> None:
    if not active:
        return
    target_logit = candidate_logits[target_offset]
    for rival_target in profile_targets:
        if rival_target == target:
            continue
        rival_offset = profile_offsets[rival_target]
        margin_gap = candidate_logits[rival_offset] - target_logit + 1.0
        losses.prompt_target_margin_loss = losses.prompt_target_margin_loss + (
            Scalar(1.0) + margin_gap.exp()
        ).log()
        losses.prompt_target_margin_count += 1


def add_deficit_target_loss(
    losses: BranchContextLossState,
    target_state: BranchContextReplayTargetState,
    profile_key: str,
    target: int,
    target_offset: int,
    candidate_probs: list[Scalar],
    *,
    active: bool,
) -> None:
    if not active:
        return
    target_key = (profile_key, target)
    target_deficit_loss = -(candidate_probs[target_offset] + 1e-12).log()
    losses.deficit_target_loss = losses.deficit_target_loss + target_deficit_loss
    losses.deficit_target_count += 1
    losses.deficit_target_losses_by_target[target_key] = (
        losses.deficit_target_losses_by_target.get(target_key, Scalar(0.0))
        + target_deficit_loss
    )
    losses.deficit_target_counts_by_target[target_key] += 1


def add_coverage_preservation_loss(
    losses: BranchContextLossState,
    profile_key: str,
    predicted: int,
    profile_offsets: dict[int, int],
    candidate_probs: list[Scalar],
    *,
    active: bool,
) -> None:
    if not (active and predicted in profile_offsets):
        return
    predicted_key = (profile_key, predicted)
    predicted_offset = profile_offsets[predicted]
    preservation_loss = -(candidate_probs[predicted_offset] + 1e-12).log()
    losses.coverage_preservation_losses_by_target[predicted_key] = (
        losses.coverage_preservation_losses_by_target.get(predicted_key, Scalar(0.0))
        + preservation_loss
    )
    losses.coverage_preservation_counts_by_target[predicted_key] += 1


def add_covered_anchor_loss(
    losses: BranchContextLossState,
    profile_key: str,
    target: int,
    predicted: int,
    target_offset: int,
    candidate_probs: list[Scalar],
    *,
    active: bool,
) -> None:
    if not active:
        return
    target_key = (profile_key, target)
    target_anchor_loss = -(candidate_probs[target_offset] + 1e-12).log()
    losses.covered_anchor_loss = losses.covered_anchor_loss + target_anchor_loss
    losses.covered_anchor_count += 1
    losses.covered_anchor_losses_by_target[target_key] = (
        losses.covered_anchor_losses_by_target.get(target_key, Scalar(0.0))
        + target_anchor_loss
    )
    losses.covered_anchor_counts_by_target[target_key] += 1
