"""Memory-consolidation fields for direct-answer update-guard state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import transformer_direct_modes as modes


def build_memory_consolidation_guard_state(
    *,
    flags: dict[str, bool],
    memory_consolidation_max_profiles: int,
    direct_memory_consolidation_source_plan_path: Path | None,
    direct_memory_consolidation_source_plan_summary: dict[str, Any],
    direct_memory_consolidation_target_profiles: list[str],
    direct_memory_consolidation_top_priority_profiles: list[str],
    direct_memory_consolidation_collapsed_memory_backed_profiles: list[str],
    direct_memory_consolidation_missing_first_token_values: dict[str, list[str]],
    direct_memory_consolidation_missing_first_token_ids: dict[str, list[int]],
    direct_memory_consolidation_profile_specific_missing_first_token_target_map: dict[
        str, list[str]
    ],
) -> dict[str, Any]:
    remaining_collapsed_active = _remaining_collapsed_active(flags)
    profile_specific_missing_token_active = _profile_specific_missing_token_active(
        flags
    )
    target_profile_set = set(direct_memory_consolidation_target_profiles)
    return {
        "profile_scale_memory_consolidation_missing_first_token_learning_rate_scales": (
            list(modes.BASELINE_FLOOR_MISSING_FIRST_TOKEN_LEARNING_RATE_SCALES)
            if flags[
                "profile_scale_memory_consolidation_missing_first_token_frontier_stabilization_active"
            ]
            else []
        ),
        "profile_scale_memory_consolidation_source_plan_path": (
            str(direct_memory_consolidation_source_plan_path)
            if direct_memory_consolidation_source_plan_path is not None
            else None
        ),
        "profile_scale_memory_consolidation_source_plan_summary": (
            direct_memory_consolidation_source_plan_summary
        ),
        "profile_scale_memory_consolidation_target_profiles": (
            direct_memory_consolidation_target_profiles
        ),
        "profile_scale_memory_consolidation_top_priority_profiles": (
            direct_memory_consolidation_top_priority_profiles
        ),
        "profile_scale_memory_consolidation_collapsed_memory_backed_profiles": (
            direct_memory_consolidation_collapsed_memory_backed_profiles
        ),
        "profile_scale_memory_consolidation_max_profiles": (
            memory_consolidation_max_profiles
            if flags["profile_scale_memory_consolidation_frontier_stabilization_active"]
            else 0
        ),
        "profile_scale_memory_consolidation_consumed_profile_count": (
            len(direct_memory_consolidation_target_profiles)
        ),
        "profile_scale_memory_consolidation_remaining_collapsed_target_profiles": (
            list(direct_memory_consolidation_target_profiles)
            if remaining_collapsed_active
            else []
        ),
        "profile_scale_memory_consolidation_remaining_collapsed_source_profiles": (
            list(direct_memory_consolidation_collapsed_memory_backed_profiles)
            if remaining_collapsed_active
            else []
        ),
        "profile_scale_memory_consolidation_remaining_collapsed_consumed_profile_count": (
            len(direct_memory_consolidation_target_profiles)
            if remaining_collapsed_active
            else 0
        ),
        "profile_scale_memory_consolidation_remaining_collapsed_unconsumed_profiles": (
            [
                profile
                for profile in direct_memory_consolidation_collapsed_memory_backed_profiles
                if profile not in target_profile_set
            ]
            if remaining_collapsed_active
            else []
        ),
        "profile_scale_memory_consolidation_profile_specific_missing_first_token_target_map": (
            direct_memory_consolidation_profile_specific_missing_first_token_target_map
            if profile_specific_missing_token_active
            else {}
        ),
        "profile_scale_memory_consolidation_missing_first_token_target_tokens": (
            direct_memory_consolidation_missing_first_token_values
        ),
        "profile_scale_memory_consolidation_missing_first_token_target_ids": (
            direct_memory_consolidation_missing_first_token_ids
        ),
        "profile_scale_memory_consolidation_missing_first_token_unique_target_ids": (
            sorted(
                {
                    token_id
                    for token_ids in direct_memory_consolidation_missing_first_token_ids.values()
                    for token_id in token_ids
                }
            )
        ),
    }


def _remaining_collapsed_active(flags: dict[str, bool]) -> bool:
    return flags[
        "profile_scale_memory_consolidation_remaining_collapsed_missing_first_token_frontier_stabilization_active"
    ]


def _profile_specific_missing_token_active(flags: dict[str, bool]) -> bool:
    return flags[
        "profile_scale_memory_consolidation_remaining_collapsed_profile_specific_missing_first_token_frontier_stabilization_active"
    ]
