"""Profile summaries for baseline-floor replay anchors."""

from __future__ import annotations

from collections import Counter

from replay_plan import BranchReplayRecord, branch_replay_parts


def baseline_floor_anchor_profile_counts(
    anchors: list[BranchReplayRecord],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for branch in anchors:
        _context, _target, _predicted, profile = branch_replay_parts(branch)
        counts[profile] += 1
    return dict(sorted(counts.items()))


def baseline_floor_anchor_profile_groups(
    anchors: list[BranchReplayRecord],
) -> dict[str, list[BranchReplayRecord]]:
    groups: dict[str, list[BranchReplayRecord]] = {}
    for branch in anchors:
        _context, _target, _predicted, profile = branch_replay_parts(branch)
        groups.setdefault(profile, []).append(branch)
    return dict(sorted(groups.items()))


def baseline_floor_anchor_profile_target_count(
    anchors: list[BranchReplayRecord],
) -> int:
    profile_targets: set[tuple[str, int]] = set()
    for branch in anchors:
        _context, target, _predicted, profile = branch_replay_parts(branch)
        profile_targets.add((profile, target))
    return len(profile_targets)
