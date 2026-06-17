"""Initial direct-answer update-guard state."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import transformer_direct_modes as modes
from transformer_baseline_floor_anchor_profiles import (
    baseline_floor_anchor_profile_counts,
    baseline_floor_anchor_profile_target_count,
)
from transformer_direct_answer_guard_keys import (
    EMPTY_DICT_KEYS,
    EMPTY_LIST_KEYS,
    ZERO_INT_KEYS,
)
from transformer_direct_answer_guard_memory import (
    build_memory_consolidation_guard_state,
)
from transformer_direct_answer_mode_flags import direct_answer_mode_flags


def build_direct_answer_update_guard(
    *,
    direct_answer_mode: str,
    memory_consolidation_max_profiles: int,
    direct_baseline_floor_learning_rate_scales: Sequence[float],
    direct_baseline_floor_outer_learning_rate_scales: Sequence[float],
    direct_baseline_floor_repair_anchors: list[Any],
    direct_baseline_floor_frontier_anchors: list[Any],
    direct_remaining_profile_binding_target_profiles: list[str],
    direct_remaining_profile_binding_source_labels: list[str],
    direct_replay_plan: dict[str, Any] | None,
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
    flags = direct_answer_mode_flags(direct_answer_mode)
    guard: dict[str, Any] = {
        **flags,
        "profile_scale_coverage_recovery_learning_rate_scales": (
            list(modes.BASELINE_FLOOR_COVERAGE_RECOVERY_LEARNING_RATE_SCALES)
            if flags["profile_scale_coverage_recovery_frontier_stabilization_active"]
            else []
        ),
        "profile_scale_branch_diversity_recovery_learning_rate_scales": (
            list(modes.BASELINE_FLOOR_BRANCH_DIVERSITY_RECOVERY_LEARNING_RATE_SCALES)
            if flags[
                "profile_scale_branch_diversity_recovery_frontier_stabilization_active"
            ]
            else []
        ),
        "profile_scale_collapsed_profile_binding_learning_rate_scales": (
            list(modes.BASELINE_FLOOR_COLLAPSED_PROFILE_BINDING_LEARNING_RATE_SCALES)
            if flags[
                "profile_scale_collapsed_profile_binding_frontier_stabilization_active"
            ]
            else []
        ),
        "learning_rate_scales": (
            list(direct_baseline_floor_learning_rate_scales)
            if flags["adaptive"]
            else [1.0]
        ),
        "outer_learning_rate_scales": (
            list(direct_baseline_floor_outer_learning_rate_scales)
            if flags["adaptive"]
            else [1.0]
        ),
        "repair_anchor_count": len(direct_baseline_floor_repair_anchors),
        "objective_anchor_count": len(direct_baseline_floor_repair_anchors),
        "objective_anchor_batch_size": (
            modes.BASELINE_FLOOR_OBJECTIVE_ANCHOR_BATCH_SIZE
            if flags["objective_active"]
            else 0
        ),
        "objective_anchor_weight": (
            modes.BASELINE_FLOOR_OBJECTIVE_ANCHOR_WEIGHT
            if flags["objective_active"]
            else 0.0
        ),
        "stabilization_anchor_count": len(direct_baseline_floor_repair_anchors),
        "stabilization_anchor_batch_size": (
            len(direct_baseline_floor_repair_anchors)
            if (
                flags["profile_targeted_stabilization_active"]
                or flags["sequential_stabilization_active"]
            )
            else (
                modes.BASELINE_FLOOR_STABILIZATION_ANCHOR_BATCH_SIZE
                if flags["stabilization_active"]
                else 0
            )
        ),
        "stabilization_profile_target_count": (
            baseline_floor_anchor_profile_target_count(
                direct_baseline_floor_repair_anchors
            )
        ),
        "stabilization_anchor_profile_counts": (
            baseline_floor_anchor_profile_counts(direct_baseline_floor_repair_anchors)
        ),
        "stabilization_profile_group_count": len(
            baseline_floor_anchor_profile_counts(direct_baseline_floor_repair_anchors)
        ),
        "frontier_anchor_count": len(direct_baseline_floor_frontier_anchors),
        "frontier_anchor_profile_counts": (
            baseline_floor_anchor_profile_counts(direct_baseline_floor_frontier_anchors)
        ),
        "frontier_profile_group_count": len(
            baseline_floor_anchor_profile_counts(direct_baseline_floor_frontier_anchors)
        ),
        "frontier_profile_target_count": (
            baseline_floor_anchor_profile_target_count(
                direct_baseline_floor_frontier_anchors
            )
        ),
        "repair_steps_per_attempt": (
            modes.BASELINE_FLOOR_REPAIR_STEPS if flags["repair_active"] else 0
        ),
        "profile_scale_remaining_profile_binding_target_profiles": (
            direct_remaining_profile_binding_target_profiles
            if flags[
                "profile_scale_remaining_profile_binding_frontier_stabilization_active"
            ]
            else []
        ),
        "profile_scale_remaining_profile_binding_source_labels": (
            direct_remaining_profile_binding_source_labels
            if flags[
                "profile_scale_remaining_profile_binding_frontier_stabilization_active"
            ]
            else []
        ),
        "profile_scale_remaining_profile_binding_source_profiles": (
            list(
                direct_replay_plan.get("remaining_profile_binding_source_profiles", [])
            )
            if (
                flags[
                    "profile_scale_remaining_profile_binding_frontier_stabilization_active"
                ]
                and isinstance(direct_replay_plan, dict)
            )
            else []
        ),
        "profile_scale_owner_paraphrase_binding_target_profiles": (
            list(modes.BASELINE_FLOOR_OWNER_PARAPHRASE_BINDING_TARGET_PROFILES)
            if flags[
                "profile_scale_owner_paraphrase_binding_frontier_stabilization_active"
            ]
            else []
        ),
        "profile_scale_owner_paraphrase_binding_source_labels": (
            direct_remaining_profile_binding_source_labels
            if flags[
                "profile_scale_owner_paraphrase_binding_frontier_stabilization_active"
            ]
            else []
        ),
        "profile_scale_owner_paraphrase_binding_source_profiles": (
            list(direct_replay_plan.get("owner_paraphrase_binding_source_profiles", []))
            if (
                flags[
                    "profile_scale_owner_paraphrase_binding_frontier_stabilization_active"
                ]
                and isinstance(direct_replay_plan, dict)
            )
            else []
        ),
        "profile_scale_owner_paraphrase_binding_preserved_profiles": (
            list(modes.BASELINE_FLOOR_OWNER_PARAPHRASE_BINDING_PRESERVED_PROFILES)
            if flags[
                "profile_scale_owner_paraphrase_binding_frontier_stabilization_active"
            ]
            else []
        ),
        **build_memory_consolidation_guard_state(
            flags=flags,
            memory_consolidation_max_profiles=memory_consolidation_max_profiles,
            direct_memory_consolidation_source_plan_path=(
                direct_memory_consolidation_source_plan_path
            ),
            direct_memory_consolidation_source_plan_summary=(
                direct_memory_consolidation_source_plan_summary
            ),
            direct_memory_consolidation_target_profiles=(
                direct_memory_consolidation_target_profiles
            ),
            direct_memory_consolidation_top_priority_profiles=(
                direct_memory_consolidation_top_priority_profiles
            ),
            direct_memory_consolidation_collapsed_memory_backed_profiles=(
                direct_memory_consolidation_collapsed_memory_backed_profiles
            ),
            direct_memory_consolidation_missing_first_token_values=(
                direct_memory_consolidation_missing_first_token_values
            ),
            direct_memory_consolidation_missing_first_token_ids=(
                direct_memory_consolidation_missing_first_token_ids
            ),
            direct_memory_consolidation_profile_specific_missing_first_token_target_map=(
                direct_memory_consolidation_profile_specific_missing_first_token_target_map
            ),
        ),
        "calibrated_min_learning_rate_scale": (
            min(direct_baseline_floor_learning_rate_scales)
            if flags["calibrated_sequential_stabilization_active"]
            else None
        ),
        "floor_diagnostics_active": flags["active"],
        "worst_rejected_coverage_deficit": 0.0,
        "worst_rejected_coverage_violation": None,
    }
    guard.update({key: 0 for key in ZERO_INT_KEYS})
    guard.update({key: {} for key in EMPTY_DICT_KEYS})
    guard.update({key: [] for key in EMPTY_LIST_KEYS})
    return guard
