"""Replay-plan summary field names for direct-answer modes."""

from __future__ import annotations


_REPLAY_FLAG_KEYS = {
    "active": "baseline_floor_update_gate_active",
    "adaptive": "baseline_floor_adaptive_updates_active",
    "repair_active": "baseline_floor_repaired_updates_active",
    "objective_active": "baseline_floor_objective_active",
    "stabilization_active": "baseline_floor_stabilization_active",
    "profile_targeted_stabilization_active": (
        "baseline_floor_profile_targeted_stabilization_active"
    ),
    "sequential_stabilization_active": (
        "baseline_floor_sequential_stabilization_active"
    ),
    "calibrated_sequential_stabilization_active": (
        "baseline_floor_calibrated_sequential_stabilization_active"
    ),
    "profile_scale_calibrated_stabilization_active": (
        "baseline_floor_profile_scale_calibrated_stabilization_active"
    ),
    "profile_scale_diversity_stabilization_active": (
        "baseline_floor_profile_scale_diversity_stabilization_active"
    ),
    "profile_scale_frontier_stabilization_active": (
        "baseline_floor_profile_scale_frontier_stabilization_active"
    ),
    "profile_scale_coverage_frontier_stabilization_active": (
        "baseline_floor_profile_scale_coverage_frontier_stabilization_active"
    ),
    "profile_scale_coverage_prep_frontier_stabilization_active": (
        "baseline_floor_profile_scale_coverage_prep_frontier_stabilization_active"
    ),
    "profile_scale_coverage_recovery_frontier_stabilization_active": (
        "baseline_floor_profile_scale_coverage_recovery_frontier_stabilization_active"
    ),
    "profile_scale_branch_stable_coverage_recovery_frontier_stabilization_active": (
        "baseline_floor_profile_scale_branch_stable_coverage_recovery_frontier_stabilization_active"
    ),
    "profile_scale_branch_diversity_recovery_frontier_stabilization_active": (
        "baseline_floor_profile_scale_branch_diversity_recovery_frontier_stabilization_active"
    ),
    "profile_scale_collapsed_profile_binding_frontier_stabilization_active": (
        "baseline_floor_profile_scale_collapsed_profile_binding_frontier_stabilization_active"
    ),
    "profile_scale_remaining_profile_binding_frontier_stabilization_active": (
        "baseline_floor_profile_scale_remaining_profile_binding_frontier_stabilization_active"
    ),
    "profile_scale_owner_paraphrase_binding_frontier_stabilization_active": (
        "baseline_floor_profile_scale_owner_paraphrase_binding_frontier_stabilization_active"
    ),
    "profile_scale_memory_consolidation_frontier_stabilization_active": (
        "baseline_floor_profile_scale_memory_consolidation_frontier_stabilization_active"
    ),
    "profile_scale_memory_consolidation_missing_first_token_frontier_stabilization_active": (
        "baseline_floor_profile_scale_memory_consolidation_missing_first_token_frontier_stabilization_active"
    ),
    "profile_scale_memory_consolidation_remaining_collapsed_missing_first_token_frontier_stabilization_active": (
        "baseline_floor_profile_scale_memory_consolidation_remaining_collapsed_missing_first_token_frontier_stabilization_active"
    ),
    "profile_scale_memory_consolidation_remaining_collapsed_profile_specific_missing_first_token_frontier_stabilization_active": (
        "baseline_floor_profile_scale_memory_consolidation_remaining_collapsed_profile_specific_missing_first_token_frontier_stabilization_active"
    ),
}


def replay_flag_summary_fields(flags: dict[str, bool]) -> dict[str, bool]:
    return {plan_key: flags[flag] for flag, plan_key in _REPLAY_FLAG_KEYS.items()}
