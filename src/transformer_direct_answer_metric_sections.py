"""Direct-answer metrics section builders."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _guard_flag(direct_answer_update_guard: Mapping[str, Any], key: str) -> bool:
    return bool(direct_answer_update_guard.get(key, False))


@dataclass(frozen=True)
class BaselineFloorMetricSection:
    fields: dict[str, Any]
    remaining_collapsed_memory_active: bool
    profile_specific_missing_first_token_active: bool


def build_baseline_floor_metric_section(
    direct_answer_update_guard: Mapping[str, Any],
) -> BaselineFloorMetricSection:
    remaining_collapsed_memory_active = _guard_flag(
        direct_answer_update_guard,
        "profile_scale_memory_consolidation_remaining_collapsed_missing_first_token_frontier_stabilization_active",
    )
    profile_specific_missing_first_token_active = _guard_flag(
        direct_answer_update_guard,
        "profile_scale_memory_consolidation_remaining_collapsed_profile_specific_missing_first_token_frontier_stabilization_active",
    )
    return BaselineFloorMetricSection(
        fields={
            "direct_answer_baseline_floor_update_gate_active": _guard_flag(
                direct_answer_update_guard, "active"
            ),
            "direct_answer_baseline_floor_adaptive_updates_active": _guard_flag(
                direct_answer_update_guard, "adaptive"
            ),
            "direct_answer_baseline_floor_repaired_updates_active": _guard_flag(
                direct_answer_update_guard, "repair_active"
            ),
            "direct_answer_baseline_floor_objective_active": _guard_flag(
                direct_answer_update_guard, "objective_active"
            ),
            "direct_answer_baseline_floor_stabilization_active": _guard_flag(
                direct_answer_update_guard, "stabilization_active"
            ),
            "direct_answer_baseline_floor_profile_targeted_stabilization_active": _guard_flag(
                direct_answer_update_guard, "profile_targeted_stabilization_active"
            ),
            "direct_answer_baseline_floor_sequential_stabilization_active": _guard_flag(
                direct_answer_update_guard, "sequential_stabilization_active"
            ),
            "direct_answer_baseline_floor_calibrated_sequential_stabilization_active": _guard_flag(
                direct_answer_update_guard,
                "calibrated_sequential_stabilization_active",
            ),
            "direct_answer_baseline_floor_profile_scale_calibrated_stabilization_active": _guard_flag(
                direct_answer_update_guard,
                "profile_scale_calibrated_stabilization_active",
            ),
            "direct_answer_baseline_floor_profile_scale_diversity_stabilization_active": _guard_flag(
                direct_answer_update_guard,
                "profile_scale_diversity_stabilization_active",
            ),
            "direct_answer_baseline_floor_profile_scale_frontier_stabilization_active": _guard_flag(
                direct_answer_update_guard,
                "profile_scale_frontier_stabilization_active",
            ),
            "direct_answer_baseline_floor_profile_scale_coverage_frontier_stabilization_active": _guard_flag(
                direct_answer_update_guard,
                "profile_scale_coverage_frontier_stabilization_active",
            ),
            "direct_answer_baseline_floor_profile_scale_coverage_prep_frontier_stabilization_active": _guard_flag(
                direct_answer_update_guard,
                "profile_scale_coverage_prep_frontier_stabilization_active",
            ),
            "direct_answer_baseline_floor_profile_scale_coverage_recovery_frontier_stabilization_active": _guard_flag(
                direct_answer_update_guard,
                "profile_scale_coverage_recovery_frontier_stabilization_active",
            ),
            "direct_answer_baseline_floor_profile_scale_branch_stable_coverage_recovery_frontier_stabilization_active": _guard_flag(
                direct_answer_update_guard,
                "profile_scale_branch_stable_coverage_recovery_frontier_stabilization_active",
            ),
            "direct_answer_baseline_floor_profile_scale_branch_diversity_recovery_frontier_stabilization_active": _guard_flag(
                direct_answer_update_guard,
                "profile_scale_branch_diversity_recovery_frontier_stabilization_active",
            ),
            "direct_answer_baseline_floor_profile_scale_collapsed_profile_binding_frontier_stabilization_active": _guard_flag(
                direct_answer_update_guard,
                "profile_scale_collapsed_profile_binding_frontier_stabilization_active",
            ),
            "direct_answer_baseline_floor_profile_scale_remaining_profile_binding_frontier_stabilization_active": _guard_flag(
                direct_answer_update_guard,
                "profile_scale_remaining_profile_binding_frontier_stabilization_active",
            ),
            "direct_answer_baseline_floor_profile_scale_owner_paraphrase_binding_frontier_stabilization_active": _guard_flag(
                direct_answer_update_guard,
                "profile_scale_owner_paraphrase_binding_frontier_stabilization_active",
            ),
            "direct_answer_baseline_floor_profile_scale_memory_consolidation_frontier_stabilization_active": _guard_flag(
                direct_answer_update_guard,
                "profile_scale_memory_consolidation_frontier_stabilization_active",
            ),
            "direct_answer_baseline_floor_profile_scale_missing_first_token_consolidation_frontier_stabilization_active": _guard_flag(
                direct_answer_update_guard,
                "profile_scale_memory_consolidation_missing_first_token_frontier_stabilization_active",
            ),
            "direct_answer_baseline_floor_profile_scale_remaining_collapsed_missing_first_token_consolidation_frontier_stabilization_active": (
                remaining_collapsed_memory_active
            ),
            "direct_answer_baseline_floor_profile_scale_remaining_collapsed_profile_specific_missing_first_token_consolidation_frontier_stabilization_active": (
                profile_specific_missing_first_token_active
            ),
        },
        remaining_collapsed_memory_active=remaining_collapsed_memory_active,
        profile_specific_missing_first_token_active=(
            profile_specific_missing_first_token_active
        ),
    )


def build_memory_consolidation_metric_section(
    *,
    source_plan_path: Path | None,
    target_profiles: list[str],
    top_priority_profiles: list[str],
    collapsed_memory_backed_profiles: list[str],
    missing_first_token_values: dict[str, list[str]],
    missing_first_token_ids: dict[str, list[int]],
    profile_specific_missing_first_token_target_map: dict[str, list[str]],
    remaining_collapsed_memory_active: bool,
    profile_specific_missing_first_token_active: bool,
) -> dict[str, Any]:
    return {
        "direct_answer_memory_consolidation_source_plan": (
            str(source_plan_path) if source_plan_path is not None else None
        ),
        "direct_answer_memory_consolidation_target_profiles": target_profiles,
        "direct_answer_memory_consolidation_top_priority_profiles": (
            top_priority_profiles
        ),
        "direct_answer_memory_consolidation_collapsed_memory_backed_profiles": (
            collapsed_memory_backed_profiles
        ),
        "direct_answer_memory_consolidation_missing_first_token_target_tokens": (
            missing_first_token_values
        ),
        "direct_answer_memory_consolidation_missing_first_token_target_ids": (
            missing_first_token_ids
        ),
        "direct_answer_memory_consolidation_remaining_collapsed_target_profiles": (
            list(target_profiles) if remaining_collapsed_memory_active else []
        ),
        "direct_answer_memory_consolidation_profile_specific_missing_first_token_target_map": (
            profile_specific_missing_first_token_target_map
            if profile_specific_missing_first_token_active
            else {}
        ),
    }
