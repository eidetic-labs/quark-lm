from __future__ import annotations

SAMPLE_KEYS = (
    "sequential_profile_probe_sample",
    "profile_scale_probe_sample",
    "profile_scale_diversity_probe_sample",
    "profile_scale_frontier_probe_sample",
    "profile_scale_coverage_frontier_probe_sample",
    "profile_scale_coverage_prep_frontier_probe_sample",
    "profile_scale_coverage_recovery_frontier_probe_sample",
    "profile_scale_branch_stable_coverage_recovery_frontier_probe_sample",
    "profile_scale_branch_diversity_recovery_frontier_probe_sample",
    "profile_scale_collapsed_profile_binding_frontier_probe_sample",
    "profile_scale_remaining_profile_binding_probe_sample",
    "profile_scale_owner_paraphrase_binding_probe_sample",
    "profile_scale_memory_consolidation_probe_sample",
    "profile_scale_memory_consolidation_missing_first_token_probe_sample",
)

BASE_SAMPLE_KEYS = SAMPLE_KEYS[:6]
OPTIONAL_SAMPLE_KEYS = SAMPLE_KEYS[6:]


def empty_guard() -> dict[str, list[dict[str, object]]]:
    return {key: [] for key in SAMPLE_KEYS}
