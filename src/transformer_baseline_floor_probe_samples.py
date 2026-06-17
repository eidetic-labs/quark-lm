"""Generic baseline-floor probe sample routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SAMPLE_LIMIT = 12


@dataclass(frozen=True)
class BaselineFloorProbeSampleStreams:
    coverage_recovery: bool = False
    branch_stable_coverage_recovery: bool = False
    branch_diversity_recovery: bool = False
    collapsed_profile_binding: bool = False
    remaining_profile_binding: bool = False
    owner_paraphrase_binding: bool = False
    memory_consolidation: bool = False
    missing_first_token: bool = False


_BASE_SAMPLE_KEYS = (
    "sequential_profile_probe_sample",
    "profile_scale_probe_sample",
    "profile_scale_diversity_probe_sample",
    "profile_scale_frontier_probe_sample",
    "profile_scale_coverage_frontier_probe_sample",
    "profile_scale_coverage_prep_frontier_probe_sample",
)


def append_sample(
    update_guard: dict[str, Any],
    key: str,
    sample: dict[str, Any],
) -> None:
    bucket = update_guard[key]
    if isinstance(bucket, list) and len(bucket) < SAMPLE_LIMIT:
        bucket.append(sample)


def increment_sample_count(update_guard: dict[str, Any], key: str, item: str) -> None:
    counts = update_guard[key]
    if isinstance(counts, dict):
        counts[item] = int(counts.get(item, 0)) + 1


def append_baseline_floor_probe_sample(
    update_guard: dict[str, Any],
    sample: dict[str, Any],
    streams: BaselineFloorProbeSampleStreams | None = None,
) -> None:
    streams = streams or BaselineFloorProbeSampleStreams()
    for key in _BASE_SAMPLE_KEYS:
        append_sample(update_guard, key, sample)
    if streams.coverage_recovery:
        append_sample(
            update_guard,
            "profile_scale_coverage_recovery_frontier_probe_sample",
            sample,
        )
    if streams.branch_stable_coverage_recovery:
        append_sample(
            update_guard,
            "profile_scale_branch_stable_coverage_recovery_frontier_probe_sample",
            sample,
        )
    if streams.branch_diversity_recovery:
        append_sample(
            update_guard,
            "profile_scale_branch_diversity_recovery_frontier_probe_sample",
            sample,
        )
    if streams.collapsed_profile_binding:
        append_sample(
            update_guard,
            "profile_scale_collapsed_profile_binding_frontier_probe_sample",
            sample,
        )
    if streams.remaining_profile_binding:
        append_sample(
            update_guard,
            "profile_scale_remaining_profile_binding_probe_sample",
            sample,
        )
    if streams.owner_paraphrase_binding:
        append_sample(
            update_guard,
            "profile_scale_owner_paraphrase_binding_probe_sample",
            sample,
        )
    if streams.memory_consolidation:
        append_sample(
            update_guard,
            "profile_scale_memory_consolidation_probe_sample",
            sample,
        )
    if streams.missing_first_token:
        append_sample(
            update_guard,
            "profile_scale_memory_consolidation_missing_first_token_probe_sample",
            sample,
        )
