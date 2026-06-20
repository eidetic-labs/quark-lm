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
    for branch in branches:
        _context, target, predicted, profile = branch_replay_parts(branch)
        profiles[profile] += 1
        targets[target] += 1
        predictions[predicted] += 1
        if predicted == target:
            represented_targets[target] += 1
    for anchor in retention_anchors:
        _context, _target, _predicted, profile = branch_replay_parts(anchor)
        retention_profiles[profile] += 1
    for anchor in target_floor_anchors:
        _context, _target, _predicted, profile = branch_replay_parts(anchor)
        target_floor_profiles[profile] += 1
    return {
        "step": direct_step,
        "min_targets_per_profile": min_targets_per_profile,
        "branch_count": len(branches),
        "profiles": sorted(profiles),
        "profile_counts": dict(sorted(profiles.items())),
        "target_count": len(targets),
        "predicted_count": len(predictions),
        "represented_target_count": len(represented_targets),
        "retention_anchor_count": len(retention_anchors),
        "retention_anchor_profile_counts": dict(sorted(retention_profiles.items())),
        "target_floor_anchor_count": len(target_floor_anchors),
        "target_floor_anchor_profile_counts": dict(
            sorted(target_floor_profiles.items())
        ),
        "target_floor_rank_summary": target_floor_rank_summary,
        "target_floor_competitor_summary": target_floor_competitor_summary,
    }
