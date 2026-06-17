"""Batch selection for baseline-floor coverage recovery."""

from __future__ import annotations

from replay_plan import BranchReplayRecord, branch_replay_parts


def select_coverage_recovery_batch(
    profile_batch: list[BranchReplayRecord],
    frontier_targets_by_profile: dict[str, set[int]],
    profile: str,
) -> list[BranchReplayRecord]:
    recovery_frontier_targets = frontier_targets_by_profile.get(profile, set())
    recovery_batch = [
        branch
        for branch in profile_batch
        if branch_replay_parts(branch)[1] in recovery_frontier_targets
    ]
    if recovery_batch:
        return recovery_batch
    return profile_batch
