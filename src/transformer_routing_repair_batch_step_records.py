"""Step-level evidence records for routing-repair batches."""

from __future__ import annotations

from collections import Counter
from typing import Any

from replay_plan import branch_replay_parts


def routing_repair_batch_step_record(
    direct_step: int,
    branches: list[Any],
    retention_anchors: list[Any],
    target_floor_anchors: list[Any],
    min_targets_per_profile: int,
    target_floor_rank_summary: dict[str, Any] | None = None,
    target_floor_competitor_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profiles: Counter[str] = Counter()
    targets: Counter[int] = Counter()
    predictions: Counter[int] = Counter()
    represented_targets: Counter[int] = Counter()
    retention_profiles: Counter[str] = Counter()
    target_floor_profiles: Counter[str] = Counter()
    target_counts_by_profile: dict[str, Counter[int]] = {}
    predicted_counts_by_profile: dict[str, Counter[int]] = {}
    represented_target_counts_by_profile: dict[str, Counter[int]] = {}
    target_floor_target_counts_by_profile: dict[str, Counter[int]] = {}
    for branch in branches:
        _context, target, predicted, profile = branch_replay_parts(branch)
        profiles[profile] += 1
        targets[target] += 1
        predictions[predicted] += 1
        _increment_nested(target_counts_by_profile, profile, target)
        _increment_nested(predicted_counts_by_profile, profile, predicted)
        if predicted == target:
            represented_targets[target] += 1
            _increment_nested(represented_target_counts_by_profile, profile, target)
    for anchor in retention_anchors:
        _context, _target, _predicted, profile = branch_replay_parts(anchor)
        retention_profiles[profile] += 1
    for anchor in target_floor_anchors:
        _context, target, _predicted, profile = branch_replay_parts(anchor)
        target_floor_profiles[profile] += 1
        _increment_nested(target_floor_target_counts_by_profile, profile, target)
    return {
        "step": direct_step,
        "min_targets_per_profile": min_targets_per_profile,
        "branch_count": len(branches),
        "profiles": sorted(profiles),
        "profile_counts": dict(sorted(profiles.items())),
        "target_counts_by_profile": _nested_counter_to_dict(
            target_counts_by_profile
        ),
        "predicted_counts_by_profile": _nested_counter_to_dict(
            predicted_counts_by_profile
        ),
        "represented_target_counts_by_profile": _nested_counter_to_dict(
            represented_target_counts_by_profile
        ),
        "target_count": len(targets),
        "predicted_count": len(predictions),
        "represented_target_count": len(represented_targets),
        "retention_anchor_count": len(retention_anchors),
        "retention_anchor_profile_counts": dict(sorted(retention_profiles.items())),
        "target_floor_anchor_count": len(target_floor_anchors),
        "target_floor_anchor_profile_counts": dict(
            sorted(target_floor_profiles.items())
        ),
        "target_floor_anchor_target_counts_by_profile": _nested_counter_to_dict(
            target_floor_target_counts_by_profile
        ),
        "target_floor_rank_summary": target_floor_rank_summary,
        "target_floor_competitor_summary": target_floor_competitor_summary,
    }


def _increment_nested(
    counters: dict[str, Counter[int]],
    profile: str,
    token_id: int,
) -> None:
    counters.setdefault(profile, Counter())[token_id] += 1


def _nested_counter_to_dict(
    counters: dict[str, Counter[int]],
) -> dict[str, dict[str, int]]:
    return {
        profile: {
            str(token_id): count
            for token_id, count in sorted(counter.items())
        }
        for profile, counter in sorted(counters.items())
    }
