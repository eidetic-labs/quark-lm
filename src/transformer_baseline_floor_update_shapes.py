"""Baseline-floor update-shape labels for guarded direct-answer updates."""

from __future__ import annotations

from transformer_direct_answer_mode_flags import direct_answer_mode_flags


def baseline_floor_attempt_update_shape(direct_answer_mode: str) -> str:
    flags = direct_answer_mode_flags(direct_answer_mode)
    if flags["profile_scale_calibrated_stabilization_active"]:
        return _profile_scale_update_shape(flags)
    if flags["calibrated_sequential_stabilization_active"]:
        return "calibrated_sequential_profile_stabilization"
    if flags["sequential_stabilization_active"]:
        return "sequential_profile_stabilization"
    if flags["profile_targeted_stabilization_active"]:
        return "profile_targeted_stabilization"
    if flags["stabilization_active"]:
        return "stabilization"
    return "direct"


def _profile_scale_update_shape(flags: dict[str, bool]) -> str:
    if flags["profile_scale_branch_diversity_recovery_frontier_stabilization_active"]:
        return _branch_diversity_recovery_update_shape(flags)
    if flags[
        "profile_scale_branch_stable_coverage_recovery_frontier_stabilization_active"
    ]:
        return (
            "profile_scale_branch_stable_coverage_recovery_frontier_diversity_"
            "calibrated_sequential_profile_stabilization"
        )
    if flags["profile_scale_coverage_recovery_frontier_stabilization_active"]:
        return (
            "profile_scale_coverage_recovery_frontier_diversity_"
            "calibrated_sequential_profile_stabilization"
        )
    if flags["profile_scale_coverage_prep_frontier_stabilization_active"]:
        return (
            "profile_scale_coverage_prep_frontier_diversity_"
            "calibrated_sequential_profile_stabilization"
        )
    if flags["profile_scale_coverage_frontier_stabilization_active"]:
        return (
            "profile_scale_coverage_frontier_diversity_"
            "calibrated_sequential_profile_stabilization"
        )
    if flags["profile_scale_frontier_stabilization_active"]:
        return (
            "profile_scale_frontier_diversity_"
            "calibrated_sequential_profile_stabilization"
        )
    if flags["profile_scale_diversity_stabilization_active"]:
        return (
            "profile_scale_diversity_calibrated_sequential_profile_stabilization"
        )
    return "profile_scale_calibrated_sequential_profile_stabilization"


def _branch_diversity_recovery_update_shape(flags: dict[str, bool]) -> str:
    if not flags["profile_scale_collapsed_profile_binding_frontier_stabilization_active"]:
        return (
            "profile_scale_branch_diversity_recovery_frontier_"
            "calibrated_sequential_profile_stabilization"
        )
    if not flags["profile_scale_remaining_profile_binding_frontier_stabilization_active"]:
        return (
            "profile_scale_collapsed_profile_binding_frontier_"
            "calibrated_sequential_profile_stabilization"
        )
    if flags["profile_scale_memory_consolidation_frontier_stabilization_active"]:
        return _memory_consolidation_update_shape(flags)
    if flags["profile_scale_owner_paraphrase_binding_frontier_stabilization_active"]:
        return (
            "profile_scale_owner_paraphrase_binding_frontier_"
            "calibrated_sequential_profile_stabilization"
        )
    return (
        "profile_scale_remaining_profile_binding_frontier_"
        "calibrated_sequential_profile_stabilization"
    )


def _memory_consolidation_update_shape(flags: dict[str, bool]) -> str:
    if flags[
        "profile_scale_memory_consolidation_remaining_collapsed_profile_specific_missing_first_token_frontier_stabilization_active"
    ]:
        return (
            "profile_scale_memory_consolidation_remaining_collapsed_profile_specific_"
            "missing_first_token_frontier_calibrated_sequential_profile_stabilization"
        )
    if flags[
        "profile_scale_memory_consolidation_remaining_collapsed_missing_first_token_frontier_stabilization_active"
    ]:
        return (
            "profile_scale_memory_consolidation_remaining_collapsed_missing_first_token_"
            "frontier_calibrated_sequential_profile_stabilization"
        )
    if flags[
        "profile_scale_memory_consolidation_missing_first_token_frontier_stabilization_active"
    ]:
        return (
            "profile_scale_memory_consolidation_missing_first_token_frontier_"
            "calibrated_sequential_profile_stabilization"
        )
    return (
        "profile_scale_memory_consolidation_frontier_"
        "calibrated_sequential_profile_stabilization"
    )
