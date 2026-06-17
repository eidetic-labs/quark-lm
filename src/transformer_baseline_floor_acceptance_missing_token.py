"""Missing-first-token acceptance accounting."""

from __future__ import annotations

from typing import Any

from transformer_baseline_floor_acceptance_guard import set_map
from transformer_baseline_floor_acceptance_types import (
    BaselineFloorProfileAcceptanceAccounting,
)


def record_missing_first_token_outcomes(
    update_guard: dict[str, Any],
    accounting: BaselineFloorProfileAcceptanceAccounting,
) -> None:
    if accounting.missing_first_token_accepted:
        set_map(
            update_guard,
            "profile_scale_memory_consolidation_missing_first_token_profile_acceptance_outcomes",
            accounting.profile,
            "missing_first_token_coverage",
        )
    elif (
        accounting.missing_first_token_active
        and accounting.missing_first_token_attempted
    ):
        set_map(
            update_guard,
            "profile_scale_memory_consolidation_missing_first_token_profile_acceptance_outcomes",
            accounting.profile,
            "missing_first_token_fallback",
        )
    elif accounting.missing_first_token_active:
        set_map(
            update_guard,
            "profile_scale_memory_consolidation_missing_first_token_profile_acceptance_outcomes",
            accounting.profile,
            "no_missing_first_token_batch",
        )
    if accounting.missing_first_token_delta is not None:
        set_map(
            update_guard,
            "profile_scale_memory_consolidation_missing_first_token_profile_deltas",
            accounting.profile,
            {
                "target_profiles": accounting.missing_first_token_target_profiles,
                "target_ids": accounting.missing_first_token_target_ids,
                "base_score": (
                    list(accounting.missing_first_token_base_score)
                    if accounting.missing_first_token_base_score is not None
                    else None
                ),
                "final_score": (
                    list(accounting.missing_first_token_score)
                    if accounting.missing_first_token_score is not None
                    else None
                ),
                "accepted": accounting.missing_first_token_accepted,
                "outcome": accounting.missing_first_token_outcome,
                "delta": accounting.missing_first_token_delta,
            },
        )
