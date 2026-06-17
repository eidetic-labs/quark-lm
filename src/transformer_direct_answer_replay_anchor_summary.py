"""Anchor summary fields for direct-answer replay plans."""

from __future__ import annotations

from replay_plan import BranchReplayRecord
from transformer_baseline_floor_anchor_profiles import (
    baseline_floor_anchor_profile_counts,
    baseline_floor_anchor_profile_target_count,
)
import transformer_direct_modes as modes


def attach_anchor_summary(
    replay_plan: dict[str, object],
    repair_anchors: list[BranchReplayRecord],
    frontier_anchors: list[BranchReplayRecord],
    flags: dict[str, bool],
) -> None:
    replay_plan["baseline_floor_stabilization_anchor_count"] = len(repair_anchors)
    replay_plan["baseline_floor_stabilization_anchor_batch_size"] = (
        len(repair_anchors)
        if (
            flags["profile_targeted_stabilization_active"]
            or flags["sequential_stabilization_active"]
        )
        else (
            modes.BASELINE_FLOOR_STABILIZATION_ANCHOR_BATCH_SIZE
            if flags["stabilization_active"]
            else 0
        )
    )
    for prefix, anchors in (
        ("baseline_floor_stabilization", repair_anchors),
        ("baseline_floor_frontier", frontier_anchors),
    ):
        profile_counts = baseline_floor_anchor_profile_counts(anchors)
        replay_plan[f"{prefix}_profile_target_count"] = (
            baseline_floor_anchor_profile_target_count(anchors)
        )
        replay_plan[f"{prefix}_anchor_profile_counts"] = profile_counts
        replay_plan[f"{prefix}_profile_group_count"] = len(profile_counts)
    replay_plan["baseline_floor_frontier_anchor_count"] = len(frontier_anchors)
