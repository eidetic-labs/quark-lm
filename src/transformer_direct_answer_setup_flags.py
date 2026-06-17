"""Dataclass field mapping for direct-answer setup mode flags."""

from __future__ import annotations


DIRECT_ANSWER_SETUP_FLAG_FIELDS = {
    "active": "direct_answer_baseline_floor_update_gate_active",
    "adaptive": "direct_answer_baseline_floor_adaptive_updates_active",
    "repair_active": "direct_answer_baseline_floor_repaired_updates_active",
    "objective_active": "direct_answer_baseline_floor_objective_active",
    "stabilization_active": "direct_answer_baseline_floor_stabilization_active",
    "profile_targeted_stabilization_active": (
        "direct_answer_baseline_floor_profile_targeted_stabilization_active"
    ),
    "sequential_stabilization_active": (
        "direct_answer_baseline_floor_sequential_stabilization_active"
    ),
    "calibrated_sequential_stabilization_active": (
        "direct_answer_baseline_floor_calibrated_sequential_stabilization_active"
    ),
    "profile_scale_calibrated_stabilization_active": (
        "direct_answer_baseline_floor_profile_scale_calibrated_stabilization_active"
    ),
    "profile_scale_diversity_stabilization_active": (
        "direct_answer_baseline_floor_profile_scale_diversity_stabilization_active"
    ),
    "profile_scale_frontier_stabilization_active": (
        "direct_answer_baseline_floor_profile_scale_frontier_stabilization_active"
    ),
    "profile_scale_coverage_frontier_stabilization_active": (
        "direct_answer_baseline_floor_profile_scale_coverage_frontier_stabilization_active"
    ),
    "profile_scale_coverage_prep_frontier_stabilization_active": (
        "direct_answer_baseline_floor_profile_scale_coverage_prep_frontier_stabilization_active"
    ),
    "profile_scale_coverage_recovery_frontier_stabilization_active": (
        "direct_answer_baseline_floor_profile_scale_coverage_recovery_frontier_stabilization_active"
    ),
    "profile_scale_branch_stable_coverage_recovery_frontier_stabilization_active": (
        "direct_answer_baseline_floor_profile_scale_branch_stable_coverage_recovery_frontier_stabilization_active"
    ),
    "profile_scale_branch_diversity_recovery_frontier_stabilization_active": (
        "direct_answer_baseline_floor_profile_scale_branch_diversity_recovery_frontier_stabilization_active"
    ),
    "profile_scale_collapsed_profile_binding_frontier_stabilization_active": (
        "direct_answer_baseline_floor_profile_scale_collapsed_profile_binding_frontier_stabilization_active"
    ),
    "profile_scale_remaining_profile_binding_frontier_stabilization_active": (
        "direct_answer_baseline_floor_profile_scale_remaining_profile_binding_frontier_stabilization_active"
    ),
    "profile_scale_owner_paraphrase_binding_frontier_stabilization_active": (
        "direct_answer_baseline_floor_profile_scale_owner_paraphrase_binding_frontier_stabilization_active"
    ),
    "profile_scale_memory_consolidation_frontier_stabilization_active": (
        "direct_answer_baseline_floor_profile_scale_memory_consolidation_frontier_stabilization_active"
    ),
    "profile_scale_memory_consolidation_missing_first_token_frontier_stabilization_active": (
        "direct_answer_baseline_floor_profile_scale_missing_first_token_consolidation_frontier_stabilization_active"
    ),
    "profile_scale_memory_consolidation_remaining_collapsed_missing_first_token_frontier_stabilization_active": (
        "direct_answer_baseline_floor_profile_scale_remaining_collapsed_missing_first_token_consolidation_frontier_stabilization_active"
    ),
    "profile_scale_memory_consolidation_remaining_collapsed_profile_specific_missing_first_token_frontier_stabilization_active": (
        "direct_answer_baseline_floor_profile_scale_remaining_collapsed_profile_specific_missing_first_token_consolidation_frontier_stabilization_active"
    ),
}


def direct_answer_setup_flag_field_kwargs(flags: dict[str, bool]) -> dict[str, bool]:
    return {
        field: flags[flag]
        for flag, field in DIRECT_ANSWER_SETUP_FLAG_FIELDS.items()
    }
