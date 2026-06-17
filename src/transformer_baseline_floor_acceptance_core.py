"""Core acceptance counters for baseline-floor profile-scale updates."""

from __future__ import annotations

from typing import Any

from transformer_baseline_floor_acceptance_branch_diversity import (
    record_branch_diversity_outcomes,
)
from transformer_baseline_floor_acceptance_collapsed_binding import (
    record_collapsed_binding_outcomes,
)
from transformer_baseline_floor_acceptance_guard import (
    increment,
    increment_map,
    set_map,
)
from transformer_baseline_floor_acceptance_missing_token import (
    record_missing_first_token_outcomes,
)
from transformer_baseline_floor_acceptance_recovery import record_recovery_outcomes
from transformer_baseline_floor_acceptance_types import (
    BaselineFloorProfileAcceptanceAccounting,
)


def record_baseline_floor_profile_acceptance(
    update_guard: dict[str, Any],
    accounting: BaselineFloorProfileAcceptanceAccounting,
) -> None:
    profile = accounting.profile
    increment(update_guard, "sequential_profile_acceptances")
    increment(update_guard, "profile_scale_memory_acceptances")
    if accounting.remaining_profile_binding_prioritized:
        increment(
            update_guard,
            "profile_scale_remaining_profile_binding_prioritized_acceptances",
        )
    if accounting.owner_paraphrase_binding_prioritized:
        increment(
            update_guard,
            "profile_scale_owner_paraphrase_binding_prioritized_acceptances",
        )
    if accounting.memory_consolidation_prioritized:
        increment(
            update_guard,
            "profile_scale_memory_consolidation_prioritized_acceptances",
        )
    if accounting.diversity_active:
        increment(update_guard, "profile_scale_diversity_acceptances")
        if accounting.diversity_outcome == "improved":
            increment(update_guard, "profile_scale_diversity_score_improvements")
        else:
            increment(update_guard, "profile_scale_diversity_score_ties")
    if accounting.frontier_active:
        increment(update_guard, "profile_scale_frontier_acceptances")
    _record_coverage_counters(update_guard, accounting)
    _record_profile_maps(update_guard, accounting, profile)
    record_recovery_outcomes(update_guard, accounting)
    record_branch_diversity_outcomes(update_guard, accounting)
    record_collapsed_binding_outcomes(update_guard, accounting)
    record_missing_first_token_outcomes(update_guard, accounting)


def _record_coverage_counters(
    update_guard: dict[str, Any],
    accounting: BaselineFloorProfileAcceptanceAccounting,
) -> None:
    if accounting.coverage_frontier_active:
        increment(update_guard, "profile_scale_coverage_frontier_acceptances")
        if accounting.coverage_outcome == "gained":
            increment(update_guard, "profile_scale_coverage_frontier_gains")
        elif accounting.coverage_outcome == "tied":
            increment(update_guard, "profile_scale_coverage_frontier_ties")
    if accounting.coverage_prep_active:
        increment(update_guard, "profile_scale_coverage_prep_frontier_acceptances")
        if accounting.coverage_outcome == "gained":
            increment(
                update_guard,
                "profile_scale_coverage_prep_frontier_gain_acceptances",
            )
        elif accounting.coverage_prep_accepted:
            increment(
                update_guard,
                "profile_scale_coverage_prep_frontier_preparations",
            )
    if (
        accounting.coverage_recovery_active
        and accounting.coverage_prep_accepted
        and accounting.coverage_recovery_attempted
        and not accounting.coverage_recovery_accepted
    ):
        increment(
            update_guard,
            "profile_scale_coverage_recovery_frontier_fallback_preparations",
        )
    if (
        accounting.branch_stable_coverage_recovery_active
        and accounting.coverage_prep_accepted
        and accounting.coverage_recovery_attempted
        and not accounting.coverage_recovery_accepted
    ):
        increment(
            update_guard,
            "profile_scale_branch_stable_coverage_recovery_frontier_fallback_preparations",
        )


def _record_profile_maps(
    update_guard: dict[str, Any],
    accounting: BaselineFloorProfileAcceptanceAccounting,
    profile: str,
) -> None:
    increment_map(update_guard, "sequential_profile_acceptance_counts", profile)
    increment_map(
        update_guard,
        "profile_scale_acceptance_scale_counts",
        accounting.scale_key,
    )
    set_map(
        update_guard,
        "profile_scale_profile_acceptance_scales",
        profile,
        accounting.scale_key,
    )
    set_map(
        update_guard,
        "profile_scale_diversity_profile_acceptance_outcomes",
        profile,
        accounting.diversity_outcome,
    )
    if accounting.coverage_delta is not None:
        set_map(
            update_guard,
            "profile_scale_coverage_frontier_profile_acceptance_deltas",
            profile,
            accounting.coverage_delta,
        )
    set_map(
        update_guard,
        "profile_scale_coverage_prep_frontier_profile_acceptance_outcomes",
        profile,
        "coverage_gain"
        if accounting.coverage_outcome == "gained"
        else "coverage_preparation",
    )
