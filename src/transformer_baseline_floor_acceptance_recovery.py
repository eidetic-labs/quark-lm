"""Coverage-recovery acceptance accounting for baseline-floor updates."""

from __future__ import annotations

from typing import Any

from transformer_baseline_floor_acceptance_guard import set_map
from transformer_baseline_floor_acceptance_types import (
    BaselineFloorProfileAcceptanceAccounting,
)


def record_recovery_outcomes(
    update_guard: dict[str, Any],
    accounting: BaselineFloorProfileAcceptanceAccounting,
) -> None:
    if accounting.coverage_recovery_accepted:
        set_map(
            update_guard,
            "profile_scale_coverage_recovery_frontier_profile_acceptance_outcomes",
            accounting.profile,
            "coverage_recovery",
        )
    elif accounting.coverage_recovery_attempted:
        set_map(
            update_guard,
            "profile_scale_coverage_recovery_frontier_profile_acceptance_outcomes",
            accounting.profile,
            "coverage_preparation_fallback",
        )
    elif accounting.coverage_outcome == "gained":
        set_map(
            update_guard,
            "profile_scale_coverage_recovery_frontier_profile_acceptance_outcomes",
            accounting.profile,
            "coverage_gain",
        )

    if accounting.coverage_recovery_branch_stable_accepted:
        set_map(
            update_guard,
            "profile_scale_branch_stable_coverage_recovery_frontier_profile_acceptance_outcomes",
            accounting.profile,
            "branch_stable_coverage_recovery",
        )
    elif (
        accounting.branch_stable_coverage_recovery_active
        and accounting.coverage_recovery_attempted
    ):
        set_map(
            update_guard,
            "profile_scale_branch_stable_coverage_recovery_frontier_profile_acceptance_outcomes",
            accounting.profile,
            "branch_stable_preparation_fallback",
        )
    elif (
        accounting.branch_stable_coverage_recovery_active
        and accounting.coverage_outcome == "gained"
    ):
        set_map(
            update_guard,
            "profile_scale_branch_stable_coverage_recovery_frontier_profile_acceptance_outcomes",
            accounting.profile,
            "coverage_gain",
        )
