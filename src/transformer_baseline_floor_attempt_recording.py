"""Acceptance-record assembly for profile-scale baseline-floor attempts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from transformer_baseline_floor_acceptance_routing import (
    BaselineFloorProfileAcceptanceAttempt,
    record_baseline_floor_profile_attempt_acceptance,
)
from transformer_baseline_floor_attempt_state import (
    BaselineFloorProfileAttemptState,
)


@dataclass(frozen=True)
class BaselineFloorProfileAcceptanceContext:
    profile: str
    records: int
    frontier_records: int
    learning_rate_scale: float
    scale_key: str
    remaining_profile_binding_active: bool
    remaining_profile_binding_prioritized: bool
    remaining_profile_binding_target_profiles: Sequence[str]
    remaining_profile_binding_source_profiles: Sequence[str]
    owner_paraphrase_binding_active: bool
    owner_paraphrase_binding_prioritized: bool
    owner_paraphrase_binding_target_profiles: Sequence[str]
    owner_paraphrase_binding_preserved_profiles: Sequence[str]
    memory_consolidation_active: bool
    memory_consolidation_prioritized: bool
    memory_consolidation_target_profiles: Sequence[str]
    memory_consolidation_source_plan: str
    memory_consolidation_collapsed_memory_backed_profiles: Sequence[str]
    memory_consolidation_remaining_collapsed_active: bool
    memory_consolidation_profile_specific_active: bool
    memory_consolidation_profile_specific_missing_first_token_target_map: Any
    diversity_active: bool
    frontier_active: bool
    coverage_active: bool
    coverage_frontier_active: bool
    coverage_prep_active: bool
    coverage_recovery_active: bool
    branch_stable_coverage_recovery_active: bool
    branch_diversity_recovery_active: bool
    collapsed_profile_binding_active: bool
    missing_first_token_active: bool
    missing_first_token_profile_specific: bool


def baseline_floor_profile_acceptance_attempt(
    state: BaselineFloorProfileAttemptState,
    context: BaselineFloorProfileAcceptanceContext,
) -> BaselineFloorProfileAcceptanceAttempt:
    return BaselineFloorProfileAcceptanceAttempt(
        profile=context.profile,
        records=context.records,
        frontier_records=context.frontier_records,
        learning_rate_scale=context.learning_rate_scale,
        scale_key=context.scale_key,
        remaining_profile_binding_active=context.remaining_profile_binding_active,
        remaining_profile_binding_prioritized=(
            context.remaining_profile_binding_prioritized
        ),
        remaining_profile_binding_target_profiles=list(
            context.remaining_profile_binding_target_profiles
        ),
        remaining_profile_binding_source_profiles=list(
            context.remaining_profile_binding_source_profiles
        ),
        owner_paraphrase_binding_active=context.owner_paraphrase_binding_active,
        owner_paraphrase_binding_prioritized=(
            context.owner_paraphrase_binding_prioritized
        ),
        owner_paraphrase_binding_target_profiles=list(
            context.owner_paraphrase_binding_target_profiles
        ),
        owner_paraphrase_binding_preserved_profiles=list(
            context.owner_paraphrase_binding_preserved_profiles
        ),
        owner_paraphrase_binding_preserved=(
            state.owner_paraphrase_binding_preserved
        ),
        owner_paraphrase_binding_preservation_delta=(
            state.owner_paraphrase_binding_preservation_delta
        ),
        memory_consolidation_active=context.memory_consolidation_active,
        memory_consolidation_prioritized=context.memory_consolidation_prioritized,
        memory_consolidation_target_profiles=list(
            context.memory_consolidation_target_profiles
        ),
        memory_consolidation_source_plan=context.memory_consolidation_source_plan,
        memory_consolidation_collapsed_memory_backed_profiles=list(
            context.memory_consolidation_collapsed_memory_backed_profiles
        ),
        memory_consolidation_remaining_collapsed_active=(
            context.memory_consolidation_remaining_collapsed_active
        ),
        memory_consolidation_profile_specific_active=(
            context.memory_consolidation_profile_specific_active
        ),
        memory_consolidation_profile_specific_missing_first_token_target_map=(
            context.memory_consolidation_profile_specific_missing_first_token_target_map
        ),
        diversity_active=context.diversity_active,
        diversity_outcome=state.diversity_outcome,
        profile_score=state.profile_score,
        profile_base_score=state.profile_base_score,
        frontier_active=context.frontier_active,
        coverage_active=context.coverage_active,
        coverage_frontier_active=context.coverage_frontier_active,
        coverage_outcome=state.coverage_outcome,
        coverage_prep_active=context.coverage_prep_active,
        coverage_prep_accepted=state.coverage_prep_accepted,
        coverage_delta=state.coverage_delta,
        coverage_recovery_active=context.coverage_recovery_active,
        coverage_recovery_attempted=state.coverage_recovery_attempted,
        coverage_recovery_accepted=state.coverage_recovery_accepted,
        coverage_recovery_outcome=state.coverage_recovery_outcome,
        coverage_recovery_records=state.coverage_recovery_records,
        coverage_recovery_learning_rate_scale=(
            state.coverage_recovery_learning_rate_scale
        ),
        coverage_recovery_delta=state.coverage_recovery_delta,
        branch_stable_coverage_recovery_active=(
            context.branch_stable_coverage_recovery_active
        ),
        coverage_recovery_branch_stable_checked=(
            state.coverage_recovery_branch_stable_checked
        ),
        coverage_recovery_branch_stable_accepted=(
            state.coverage_recovery_branch_stable_accepted
        ),
        coverage_recovery_branch_stability_preserved=(
            state.coverage_recovery_branch_stability_preserved
        ),
        coverage_recovery_prepared_score=state.coverage_recovery_prepared_score,
        coverage_recovery_score=state.coverage_recovery_score,
        branch_diversity_recovery_active=(
            context.branch_diversity_recovery_active
        ),
        branch_diversity_recovery_attempted=(
            state.branch_diversity_recovery_attempted
        ),
        branch_diversity_recovery_accepted=(
            state.branch_diversity_recovery_accepted
        ),
        branch_diversity_recovery_outcome=state.branch_diversity_recovery_outcome,
        branch_diversity_recovery_rejection_reason=(
            state.branch_diversity_recovery_rejection_reason
        ),
        branch_diversity_recovery_learning_rate_scale=(
            state.branch_diversity_recovery_learning_rate_scale
        ),
        branch_diversity_recovery_records=state.branch_diversity_recovery_records,
        branch_diversity_recovery_base_score=(
            state.branch_diversity_recovery_base_score
        ),
        branch_diversity_recovery_score=state.branch_diversity_recovery_score,
        branch_diversity_recovery_delta=state.branch_diversity_recovery_delta,
        collapsed_profile_binding_active=context.collapsed_profile_binding_active,
        collapsed_profile_binding_attempted=(
            state.collapsed_profile_binding_attempted
        ),
        collapsed_profile_binding_accepted=(
            state.collapsed_profile_binding_accepted
        ),
        collapsed_profile_binding_outcome=state.collapsed_profile_binding_outcome,
        collapsed_profile_binding_target_profiles=(
            state.collapsed_profile_binding_target_profiles
        ),
        collapsed_profile_binding_rejection_reason=(
            state.collapsed_profile_binding_rejection_reason
        ),
        collapsed_profile_binding_learning_rate_scale=(
            state.collapsed_profile_binding_learning_rate_scale
        ),
        collapsed_profile_binding_records=state.collapsed_profile_binding_records,
        collapsed_profile_binding_base_score=(
            state.collapsed_profile_binding_base_score
        ),
        collapsed_profile_binding_score=state.collapsed_profile_binding_score,
        collapsed_profile_binding_delta=state.collapsed_profile_binding_delta,
        missing_first_token_active=context.missing_first_token_active,
        missing_first_token_attempted=state.missing_first_token_attempted,
        missing_first_token_accepted=state.missing_first_token_accepted,
        missing_first_token_outcome=state.missing_first_token_outcome,
        missing_first_token_target_profiles=state.missing_first_token_target_profiles,
        missing_first_token_target_ids=state.missing_first_token_target_ids,
        missing_first_token_profile_specific=(
            context.missing_first_token_profile_specific
        ),
        missing_first_token_rejection_reason=(
            state.missing_first_token_rejection_reason
        ),
        missing_first_token_learning_rate_scale=(
            state.missing_first_token_learning_rate_scale
        ),
        missing_first_token_records=state.missing_first_token_records,
        missing_first_token_base_score=state.missing_first_token_base_score,
        missing_first_token_score=state.missing_first_token_score,
        missing_first_token_delta=state.missing_first_token_delta,
    )


def record_baseline_floor_profile_state_acceptance(
    update_guard: dict[str, Any],
    state: BaselineFloorProfileAttemptState,
    context: BaselineFloorProfileAcceptanceContext,
) -> None:
    record_baseline_floor_profile_attempt_acceptance(
        update_guard,
        baseline_floor_profile_acceptance_attempt(state, context),
    )
