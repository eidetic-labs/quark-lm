from __future__ import annotations

ZERO_KEYS = (
    "sequential_profile_rejections",
    "profile_scale_memory_rejections",
    "profile_scale_remaining_profile_binding_prioritized_rejections",
    "profile_scale_owner_paraphrase_binding_prioritized_rejections",
    "profile_scale_memory_consolidation_prioritized_rejections",
    "profile_scale_diversity_rejections",
    "profile_scale_diversity_floor_rejections",
    "profile_scale_diversity_score_regressions",
    "profile_scale_frontier_rejections",
    "profile_scale_coverage_frontier_rejections",
    "profile_scale_coverage_frontier_gains",
    "profile_scale_coverage_frontier_ties",
    "profile_scale_coverage_frontier_regressions",
    "profile_scale_coverage_prep_frontier_rejections",
    "profile_scale_coverage_recovery_frontier_rejections",
    "profile_scale_branch_stable_coverage_recovery_frontier_rejections",
    "profile_scale_branch_diversity_recovery_frontier_rejections",
    "profile_scale_collapsed_profile_binding_frontier_rejections",
    "profile_scale_memory_consolidation_missing_first_token_rejections",
)

MAP_KEYS = (
    "profile_scale_diversity_rejection_reasons",
    "profile_scale_coverage_frontier_rejection_reasons",
    "profile_scale_coverage_prep_frontier_rejection_reasons",
    "profile_scale_coverage_recovery_frontier_rejection_reasons",
    "profile_scale_branch_stable_coverage_recovery_frontier_rejection_reasons",
    "profile_scale_branch_diversity_recovery_frontier_rejection_reasons",
    "profile_scale_collapsed_profile_binding_frontier_rejection_reasons",
    "profile_scale_memory_consolidation_missing_first_token_rejection_reasons",
)


def empty_guard() -> dict[str, object]:
    guard: dict[str, object] = {key: 0 for key in ZERO_KEYS}
    guard.update({key: {} for key in MAP_KEYS})
    guard["sequential_profile_rejection_counts"] = {}
    guard["profile_scale_rejection_scale_counts"] = {}
    guard["sequential_profile_probe_sample"] = []
    guard["profile_scale_probe_sample"] = []
    guard["profile_scale_diversity_probe_sample"] = []
    guard["profile_scale_frontier_probe_sample"] = []
    guard["profile_scale_coverage_frontier_probe_sample"] = []
    guard["profile_scale_coverage_prep_frontier_probe_sample"] = []
    return guard
