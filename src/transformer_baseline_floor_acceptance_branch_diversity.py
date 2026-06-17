"""Branch-diversity recovery acceptance accounting."""

from __future__ import annotations

from typing import Any

from transformer_baseline_floor_acceptance_guard import set_map
from transformer_baseline_floor_acceptance_types import (
    BaselineFloorProfileAcceptanceAccounting,
)


def record_branch_diversity_outcomes(
    update_guard: dict[str, Any],
    accounting: BaselineFloorProfileAcceptanceAccounting,
) -> None:
    if accounting.branch_diversity_recovery_accepted:
        set_map(
            update_guard,
            "profile_scale_branch_diversity_recovery_frontier_profile_acceptance_outcomes",
            accounting.profile,
            "branch_diversity_recovery",
        )
    elif (
        accounting.branch_diversity_recovery_active
        and accounting.branch_diversity_recovery_attempted
    ):
        set_map(
            update_guard,
            "profile_scale_branch_diversity_recovery_frontier_profile_acceptance_outcomes",
            accounting.profile,
            "branch_diversity_fallback",
        )
    if (
        accounting.branch_diversity_recovery_base_score is not None
        and accounting.branch_diversity_recovery_score is not None
    ):
        set_map(
            update_guard,
            "profile_scale_branch_diversity_recovery_frontier_profile_score_deltas",
            accounting.profile,
            {
                "base_score": list(accounting.branch_diversity_recovery_base_score),
                "final_score": list(accounting.branch_diversity_recovery_score),
                "improved": accounting.branch_diversity_recovery_accepted,
                "outcome": accounting.branch_diversity_recovery_outcome,
            },
        )
