"""Build baseline-floor profile acceptance accounting payloads."""

from __future__ import annotations

from transformer_baseline_floor_acceptance import (
    BaselineFloorProfileAcceptanceAccounting,
)
from transformer_baseline_floor_acceptance_attempt import (
    BaselineFloorProfileAcceptanceAttempt,
)


def build_baseline_floor_profile_acceptance_accounting(
    attempt: BaselineFloorProfileAcceptanceAttempt,
) -> BaselineFloorProfileAcceptanceAccounting:
    return BaselineFloorProfileAcceptanceAccounting(
        profile=attempt.profile,
        scale_key=attempt.scale_key,
        remaining_profile_binding_prioritized=(
            attempt.remaining_profile_binding_prioritized
        ),
        owner_paraphrase_binding_prioritized=(
            attempt.owner_paraphrase_binding_prioritized
        ),
        memory_consolidation_prioritized=attempt.memory_consolidation_prioritized,
        diversity_active=attempt.diversity_active,
        diversity_outcome=attempt.diversity_outcome,
        frontier_active=attempt.frontier_active,
        coverage_frontier_active=attempt.coverage_frontier_active,
        coverage_outcome=attempt.coverage_outcome,
        coverage_delta=attempt.coverage_delta,
        coverage_prep_active=attempt.coverage_prep_active,
        coverage_prep_accepted=attempt.coverage_prep_accepted,
        coverage_recovery_active=attempt.coverage_recovery_active,
        coverage_recovery_attempted=attempt.coverage_recovery_attempted,
        coverage_recovery_accepted=attempt.coverage_recovery_accepted,
        branch_stable_coverage_recovery_active=(
            attempt.branch_stable_coverage_recovery_active
        ),
        coverage_recovery_branch_stable_accepted=(
            attempt.coverage_recovery_branch_stable_accepted
        ),
        branch_diversity_recovery_active=attempt.branch_diversity_recovery_active,
        branch_diversity_recovery_attempted=(
            attempt.branch_diversity_recovery_attempted
        ),
        branch_diversity_recovery_accepted=(
            attempt.branch_diversity_recovery_accepted
        ),
        branch_diversity_recovery_base_score=(
            attempt.branch_diversity_recovery_base_score
        ),
        branch_diversity_recovery_score=attempt.branch_diversity_recovery_score,
        branch_diversity_recovery_outcome=attempt.branch_diversity_recovery_outcome,
        collapsed_profile_binding_active=attempt.collapsed_profile_binding_active,
        collapsed_profile_binding_attempted=attempt.collapsed_profile_binding_attempted,
        collapsed_profile_binding_accepted=attempt.collapsed_profile_binding_accepted,
        collapsed_profile_binding_target_profiles=(
            attempt.collapsed_profile_binding_target_profiles
        ),
        collapsed_profile_binding_base_score=(
            attempt.collapsed_profile_binding_base_score
        ),
        collapsed_profile_binding_score=attempt.collapsed_profile_binding_score,
        collapsed_profile_binding_delta=attempt.collapsed_profile_binding_delta,
        collapsed_profile_binding_outcome=attempt.collapsed_profile_binding_outcome,
        missing_first_token_active=attempt.missing_first_token_active,
        missing_first_token_attempted=attempt.missing_first_token_attempted,
        missing_first_token_accepted=attempt.missing_first_token_accepted,
        missing_first_token_target_profiles=attempt.missing_first_token_target_profiles,
        missing_first_token_target_ids=attempt.missing_first_token_target_ids,
        missing_first_token_base_score=attempt.missing_first_token_base_score,
        missing_first_token_score=attempt.missing_first_token_score,
        missing_first_token_delta=attempt.missing_first_token_delta,
        missing_first_token_outcome=attempt.missing_first_token_outcome,
    )

