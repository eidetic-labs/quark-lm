"""Direct-answer mode flag resolution."""

from __future__ import annotations

import transformer_direct_modes as modes

_FLAG_MODE_SETS = {
    "active": modes.BASELINE_FLOOR_GATED_DIRECT_ANSWER_MODES,
    "adaptive": modes.BASELINE_FLOOR_ADAPTIVE_DIRECT_ANSWER_MODES,
    "repair_active": modes.BASELINE_FLOOR_REPAIRED_DIRECT_ANSWER_MODES,
    "objective_active": modes.BASELINE_FLOOR_OBJECTIVE_DIRECT_ANSWER_MODES,
    "stabilization_active": modes.BASELINE_FLOOR_STABILIZATION_DIRECT_ANSWER_MODES,
    "profile_targeted_stabilization_active": (
        modes.BASELINE_FLOOR_PROFILE_TARGETED_STABILIZATION_DIRECT_ANSWER_MODES
    ),
    "sequential_stabilization_active": (
        modes.BASELINE_FLOOR_SEQUENTIAL_STABILIZATION_DIRECT_ANSWER_MODES
    ),
    "calibrated_sequential_stabilization_active": (
        modes.BASELINE_FLOOR_CALIBRATED_SEQUENTIAL_STABILIZATION_DIRECT_ANSWER_MODES
    ),
    "profile_scale_calibrated_stabilization_active": (
        modes.BASELINE_FLOOR_PROFILE_SCALE_CALIBRATED_STABILIZATION_DIRECT_ANSWER_MODES
    ),
    "profile_scale_diversity_stabilization_active": (
        modes.BASELINE_FLOOR_PROFILE_SCALE_DIVERSITY_STABILIZATION_DIRECT_ANSWER_MODES
    ),
    "profile_scale_frontier_stabilization_active": (
        modes.BASELINE_FLOOR_PROFILE_SCALE_FRONTIER_STABILIZATION_DIRECT_ANSWER_MODES
    ),
    "profile_scale_coverage_frontier_stabilization_active": (
        modes.BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_FRONTIER_STABILIZATION_DIRECT_ANSWER_MODES
    ),
    "profile_scale_coverage_prep_frontier_stabilization_active": (
        modes.BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_PREP_FRONTIER_STABILIZATION_DIRECT_ANSWER_MODES
    ),
    "profile_scale_coverage_recovery_frontier_stabilization_active": (
        modes.BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_RECOVERY_FRONTIER_STABILIZATION_DIRECT_ANSWER_MODES
    ),
    "profile_scale_branch_stable_coverage_recovery_frontier_stabilization_active": (
        modes.BASELINE_FLOOR_PROFILE_SCALE_BRANCH_STABLE_COVERAGE_RECOVERY_FRONTIER_STABILIZATION_DIRECT_ANSWER_MODES
    ),
    "profile_scale_branch_diversity_recovery_frontier_stabilization_active": (
        modes.BASELINE_FLOOR_PROFILE_SCALE_BRANCH_DIVERSITY_RECOVERY_FRONTIER_STABILIZATION_DIRECT_ANSWER_MODES
    ),
    "profile_scale_collapsed_profile_binding_frontier_stabilization_active": (
        modes.BASELINE_FLOOR_PROFILE_SCALE_COLLAPSED_PROFILE_BINDING_FRONTIER_STABILIZATION_DIRECT_ANSWER_MODES
    ),
    "profile_scale_remaining_profile_binding_frontier_stabilization_active": (
        modes.BASELINE_FLOOR_PROFILE_SCALE_REMAINING_PROFILE_BINDING_FRONTIER_STABILIZATION_DIRECT_ANSWER_MODES
    ),
    "profile_scale_owner_paraphrase_binding_frontier_stabilization_active": (
        modes.BASELINE_FLOOR_PROFILE_SCALE_OWNER_PARAPHRASE_BINDING_FRONTIER_STABILIZATION_DIRECT_ANSWER_MODES
    ),
    "profile_scale_memory_consolidation_frontier_stabilization_active": (
        modes.BASELINE_FLOOR_PROFILE_SCALE_MEMORY_CONSOLIDATION_FRONTIER_STABILIZATION_DIRECT_ANSWER_MODES
    ),
    "profile_scale_memory_consolidation_missing_first_token_frontier_stabilization_active": (
        modes.BASELINE_FLOOR_PROFILE_SCALE_MISSING_FIRST_TOKEN_CONSOLIDATION_FRONTIER_STABILIZATION_DIRECT_ANSWER_MODES
    ),
    "profile_scale_memory_consolidation_remaining_collapsed_missing_first_token_frontier_stabilization_active": (
        modes.BASELINE_FLOOR_PROFILE_SCALE_REMAINING_COLLAPSED_MISSING_FIRST_TOKEN_CONSOLIDATION_FRONTIER_STABILIZATION_DIRECT_ANSWER_MODES
    ),
    "profile_scale_memory_consolidation_remaining_collapsed_profile_specific_missing_first_token_frontier_stabilization_active": (
        modes.BASELINE_FLOOR_PROFILE_SCALE_REMAINING_COLLAPSED_PROFILE_SPECIFIC_MISSING_FIRST_TOKEN_CONSOLIDATION_FRONTIER_STABILIZATION_DIRECT_ANSWER_MODES
    ),
}


def direct_answer_mode_flags(direct_answer_mode: str) -> dict[str, bool]:
    return {
        key: direct_answer_mode in mode_set
        for key, mode_set in _FLAG_MODE_SETS.items()
    }
