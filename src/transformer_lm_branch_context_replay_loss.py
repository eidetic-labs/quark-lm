"""Replay loss accumulation for branch-context objectives."""

from __future__ import annotations

from typing import Any

from transformer_lm_branch_context_loss_state import BranchContextLossState
from transformer_lm_branch_context_replay_accumulators import (
    add_coverage_preservation_loss,
    add_covered_anchor_loss,
    add_deficit_target_loss,
    add_profile_target_share_loss,
    add_prompt_target_margin_loss,
)
from transformer_lm_branch_context_replay_candidates import (
    branch_context_candidate_ids,
    branch_context_target_set_mass,
)
from transformer_lm_branch_context_targets import (
    BranchContextReplayTargetState,
    branch_context_profile_key,
)
from transformer_math import softmax_scalars


def accumulate_replay_losses(
    model: Any,
    target_state: BranchContextReplayTargetState,
    losses: BranchContextLossState,
    *,
    hard_negative_count: int,
    focus_uncovered_targets: bool,
    preserve_predicted_target_coverage: bool,
    preserve_covered_targets: bool,
    profile_aware_targets: bool,
    balance_profile_target_shares: bool,
    enforce_prompt_target_margins: bool,
) -> None:
    for context, target, predicted, profile in target_state.replay_parts:
        profile_key = branch_context_profile_key(profile, profile_aware_targets)
        profile_targets = target_state.replay_targets_by_profile.get(
            profile_key,
            target_state.replay_targets,
        )
        profile_target_set = target_state.replay_target_sets_by_profile.get(
            profile_key,
            target_state.replay_target_set,
        )
        profile_offsets = target_state.replay_target_offsets_by_profile.get(
            profile_key,
            target_state.replay_target_offsets,
        )
        if target not in profile_offsets:
            continue
        logits = model._forward_scalars(context)
        candidate_ids = branch_context_candidate_ids(
            logits,
            model.config.vocab_size,
            profile_targets,
            profile_target_set,
            hard_negative_count,
        )
        candidate_logits = [logits[candidate_id] for candidate_id in candidate_ids]
        candidate_probs = softmax_scalars(candidate_logits)
        target_set_mass = branch_context_target_set_mass(
            candidate_ids,
            candidate_probs,
            profile_target_set,
        )
        target_offset = profile_offsets[target]
        owned_target_share = candidate_probs[target_offset] / (target_set_mass + 1e-12)
        losses.replay_coverage_loss = losses.replay_coverage_loss + (
            -(target_set_mass + 1e-12).log()
        )
        losses.replay_ownership_loss = losses.replay_ownership_loss + (
            -(owned_target_share + 1e-12).log()
        )
        add_profile_target_share_loss(
            losses,
            profile_key,
            target,
            owned_target_share,
            active=(
                balance_profile_target_shares
                and profile_aware_targets
                and len(profile_targets) > 1
            ),
        )
        add_prompt_target_margin_loss(
            losses,
            target,
            profile_targets,
            profile_offsets,
            candidate_logits,
            target_offset,
            active=(
                enforce_prompt_target_margins
                and profile_aware_targets
                and len(profile_targets) > 1
            ),
        )
        add_deficit_target_loss(
            losses,
            target_state,
            profile_key,
            target,
            target_offset,
            candidate_probs,
            active=(
                focus_uncovered_targets
                and target in target_state.deficit_targets_by_profile.get(profile_key, set())
            ),
        )
        add_coverage_preservation_loss(
            losses,
            profile_key,
            predicted,
            profile_offsets,
            candidate_probs,
            active=preserve_predicted_target_coverage,
        )
        add_covered_anchor_loss(
            losses,
            profile_key,
            target,
            predicted,
            target_offset,
            candidate_probs,
            active=preserve_covered_targets and predicted == target,
        )
