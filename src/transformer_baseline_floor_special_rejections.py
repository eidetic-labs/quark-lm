"""Specialized baseline-floor rejection counters."""

from __future__ import annotations

from typing import Any

from transformer_baseline_floor_rejection_counts import (
    increment_rejection_counter,
    increment_rejection_reason,
)


def record_baseline_floor_coverage_recovery_rejection(
    update_guard: dict[str, Any],
    reason: str,
    branch_stable_active: bool = False,
) -> None:
    increment_rejection_counter(
        update_guard,
        "profile_scale_coverage_recovery_frontier_rejections",
    )
    increment_rejection_reason(
        update_guard,
        "profile_scale_coverage_recovery_frontier_rejection_reasons",
        reason,
    )
    if branch_stable_active:
        increment_rejection_counter(
            update_guard,
            "profile_scale_branch_stable_coverage_recovery_frontier_rejections",
        )
        increment_rejection_reason(
            update_guard,
            "profile_scale_branch_stable_coverage_recovery_frontier_rejection_reasons",
            reason,
        )


def record_baseline_floor_branch_diversity_recovery_rejection(
    update_guard: dict[str, Any],
    reason: str,
) -> None:
    increment_rejection_counter(
        update_guard,
        "profile_scale_branch_diversity_recovery_frontier_rejections",
    )
    increment_rejection_reason(
        update_guard,
        "profile_scale_branch_diversity_recovery_frontier_rejection_reasons",
        reason,
    )


def record_baseline_floor_collapsed_profile_binding_rejection(
    update_guard: dict[str, Any],
    reason: str,
) -> None:
    increment_rejection_counter(
        update_guard,
        "profile_scale_collapsed_profile_binding_frontier_rejections",
    )
    increment_rejection_reason(
        update_guard,
        "profile_scale_collapsed_profile_binding_frontier_rejection_reasons",
        reason,
    )


def record_baseline_floor_missing_first_token_rejection(
    update_guard: dict[str, Any],
    reason: str,
) -> None:
    increment_rejection_counter(
        update_guard,
        "profile_scale_memory_consolidation_missing_first_token_rejections",
    )
    increment_rejection_reason(
        update_guard,
        "profile_scale_memory_consolidation_missing_first_token_rejection_reasons",
        reason,
    )
