"""Collapsed-profile binding acceptance accounting."""

from __future__ import annotations

from typing import Any

from transformer_baseline_floor_acceptance_guard import set_map
from transformer_baseline_floor_acceptance_types import (
    BaselineFloorProfileAcceptanceAccounting,
)


def record_collapsed_binding_outcomes(
    update_guard: dict[str, Any],
    accounting: BaselineFloorProfileAcceptanceAccounting,
) -> None:
    if accounting.collapsed_profile_binding_accepted:
        set_map(
            update_guard,
            "profile_scale_collapsed_profile_binding_frontier_profile_acceptance_outcomes",
            accounting.profile,
            "collapsed_profile_binding",
        )
    elif (
        accounting.collapsed_profile_binding_active
        and accounting.collapsed_profile_binding_attempted
    ):
        set_map(
            update_guard,
            "profile_scale_collapsed_profile_binding_frontier_profile_acceptance_outcomes",
            accounting.profile,
            "collapsed_profile_binding_fallback",
        )
    elif (
        accounting.collapsed_profile_binding_active
        and not accounting.collapsed_profile_binding_target_profiles
    ):
        set_map(
            update_guard,
            "profile_scale_collapsed_profile_binding_frontier_profile_acceptance_outcomes",
            accounting.profile,
            "no_collapsed_profile_targets",
        )
    if accounting.collapsed_profile_binding_delta is not None:
        set_map(
            update_guard,
            "profile_scale_collapsed_profile_binding_frontier_profile_deltas",
            accounting.profile,
            {
                "target_profiles": accounting.collapsed_profile_binding_target_profiles,
                "base_score": (
                    list(accounting.collapsed_profile_binding_base_score)
                    if accounting.collapsed_profile_binding_base_score is not None
                    else None
                ),
                "final_score": (
                    list(accounting.collapsed_profile_binding_score)
                    if accounting.collapsed_profile_binding_score is not None
                    else None
                ),
                "accepted": accounting.collapsed_profile_binding_accepted,
                "outcome": accounting.collapsed_profile_binding_outcome,
                "delta": accounting.collapsed_profile_binding_delta,
            },
        )
