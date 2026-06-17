"""Profile-scale baseline-floor acceptance and rejection accounting."""

from __future__ import annotations

from typing import Any

from transformer_baseline_floor_attempt_recording import (
    BaselineFloorProfileAcceptanceContext,
    record_baseline_floor_profile_state_acceptance,
)
from transformer_baseline_floor_attempt_state import (
    BaselineFloorProfileAttemptState,
)
from transformer_baseline_floor_rejection import (
    record_baseline_floor_profile_attempt_rejection,
)
from transformer_direct_modes import (
    BASELINE_FLOOR_OWNER_PARAPHRASE_BINDING_PRESERVED_PROFILES,
    BASELINE_FLOOR_OWNER_PARAPHRASE_BINDING_TARGET_PROFILES,
)


def record_profile_scale_rejection(
    ctx: Any,
    *,
    state: BaselineFloorProfileAttemptState,
    profile: str,
    records: int,
    frontier_records: int,
    profile_scale: float,
    priorities: dict[str, bool],
    diversity_ok: bool,
    profile_base_score: tuple[float, ...] | None,
) -> None:
    setup = ctx.direct_setup
    record_baseline_floor_profile_attempt_rejection(
        ctx.update_guard,
        profile=profile,
        records=records,
        frontier_records=frontier_records,
        learning_rate_scale=profile_scale,
        scale_key=f"{profile_scale:g}",
        direct_baseline=ctx.direct_baseline,
        profile_probe_snapshot=state.profile_probe_snapshot,
        remaining_profile_binding_prioritized=priorities["remaining"],
        owner_paraphrase_binding_prioritized=priorities["owner"],
        memory_consolidation_prioritized=priorities["memory"],
        diversity_active=(
            setup.direct_answer_baseline_floor_profile_scale_diversity_stabilization_active
        ),
        floor_preserved=state.floor_preserved,
        diversity_accepted=diversity_ok,
        diversity_outcome=state.diversity_outcome,
        diversity_rejection_reason=state.diversity_rejection_reason,
        profile_score=state.profile_score,
        profile_base_score=profile_base_score,
        frontier_active=(
            setup.direct_answer_baseline_floor_profile_scale_frontier_stabilization_active
        ),
        coverage_frontier_active=(
            setup.direct_answer_baseline_floor_profile_scale_coverage_frontier_stabilization_active
        ),
        coverage_delta=state.coverage_delta,
        coverage_outcome=state.coverage_outcome,
        coverage_prep_active=(
            setup.direct_answer_baseline_floor_profile_scale_coverage_prep_frontier_stabilization_active
        ),
        coverage_prep_accepted=state.coverage_prep_accepted,
        coverage_rejection_reason=state.coverage_rejection_reason,
    )


def record_profile_scale_acceptance(
    ctx: Any,
    *,
    state: BaselineFloorProfileAttemptState,
    profile: str,
    records: int,
    frontier_records: int,
    profile_scale: float,
    priorities: dict[str, bool],
    remaining_source_profiles: list[str],
) -> None:
    setup = ctx.direct_setup
    record_baseline_floor_profile_state_acceptance(
        ctx.update_guard,
        state,
        BaselineFloorProfileAcceptanceContext(
            profile=profile,
            records=records,
            frontier_records=frontier_records,
            learning_rate_scale=profile_scale,
            scale_key=f"{profile_scale:g}",
            remaining_profile_binding_active=setup.direct_answer_baseline_floor_profile_scale_remaining_profile_binding_frontier_stabilization_active,
            remaining_profile_binding_prioritized=priorities["remaining"],
            remaining_profile_binding_target_profiles=(
                setup.direct_remaining_profile_binding_target_profiles
            ),
            remaining_profile_binding_source_profiles=remaining_source_profiles,
            owner_paraphrase_binding_active=setup.direct_answer_baseline_floor_profile_scale_owner_paraphrase_binding_frontier_stabilization_active,
            owner_paraphrase_binding_prioritized=priorities["owner"],
            owner_paraphrase_binding_target_profiles=list(
                BASELINE_FLOOR_OWNER_PARAPHRASE_BINDING_TARGET_PROFILES
            ),
            owner_paraphrase_binding_preserved_profiles=list(
                BASELINE_FLOOR_OWNER_PARAPHRASE_BINDING_PRESERVED_PROFILES
            ),
            memory_consolidation_active=setup.direct_answer_baseline_floor_profile_scale_memory_consolidation_frontier_stabilization_active,
            memory_consolidation_prioritized=priorities["memory"],
            memory_consolidation_target_profiles=(
                setup.direct_memory_consolidation_target_profiles
            ),
            memory_consolidation_source_plan=str(
                setup.direct_memory_consolidation_source_plan_path
            ),
            memory_consolidation_collapsed_memory_backed_profiles=(
                setup.direct_memory_consolidation_collapsed_memory_backed_profiles
            ),
            memory_consolidation_remaining_collapsed_active=setup.direct_answer_baseline_floor_profile_scale_remaining_collapsed_missing_first_token_consolidation_frontier_stabilization_active,
            memory_consolidation_profile_specific_active=setup.direct_answer_baseline_floor_profile_scale_remaining_collapsed_profile_specific_missing_first_token_consolidation_frontier_stabilization_active,
            memory_consolidation_profile_specific_missing_first_token_target_map=(
                setup.direct_memory_consolidation_profile_specific_missing_first_token_target_map
            ),
            diversity_active=setup.direct_answer_baseline_floor_profile_scale_diversity_stabilization_active,
            frontier_active=setup.direct_answer_baseline_floor_profile_scale_frontier_stabilization_active,
            coverage_active=setup.direct_answer_baseline_floor_profile_scale_coverage_frontier_stabilization_active,
            coverage_frontier_active=setup.direct_answer_baseline_floor_profile_scale_coverage_frontier_stabilization_active,
            coverage_prep_active=setup.direct_answer_baseline_floor_profile_scale_coverage_prep_frontier_stabilization_active,
            coverage_recovery_active=setup.direct_answer_baseline_floor_profile_scale_coverage_recovery_frontier_stabilization_active,
            branch_stable_coverage_recovery_active=setup.direct_answer_baseline_floor_profile_scale_branch_stable_coverage_recovery_frontier_stabilization_active,
            branch_diversity_recovery_active=setup.direct_answer_baseline_floor_profile_scale_branch_diversity_recovery_frontier_stabilization_active,
            collapsed_profile_binding_active=setup.direct_answer_baseline_floor_profile_scale_collapsed_profile_binding_frontier_stabilization_active,
            missing_first_token_active=setup.direct_answer_baseline_floor_profile_scale_missing_first_token_consolidation_frontier_stabilization_active,
            missing_first_token_profile_specific=setup.direct_answer_baseline_floor_profile_scale_remaining_collapsed_profile_specific_missing_first_token_consolidation_frontier_stabilization_active,
        ),
    )
