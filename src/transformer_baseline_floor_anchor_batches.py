"""Baseline-floor anchor record construction and sampling."""

from __future__ import annotations

import random

from replay_plan import BranchReplayRecord, branch_replay_parts


def baseline_floor_repair_anchor_records(
    replay_records: list[BranchReplayRecord],
) -> list[BranchReplayRecord]:
    targets_by_profile: dict[str, set[int]] = {}
    for branch in replay_records:
        _context, target, _predicted, profile = branch_replay_parts(branch)
        targets_by_profile.setdefault(profile, set()).add(target)
    anchors: list[BranchReplayRecord] = []
    seen: set[tuple[tuple[int, ...], int, str]] = set()
    for branch in replay_records:
        context, _target, predicted, profile = branch_replay_parts(branch)
        if predicted not in targets_by_profile.get(profile, set()):
            continue
        key = (tuple(context), predicted, profile)
        if key in seen:
            continue
        seen.add(key)
        anchors.append((context, predicted, predicted, profile))
    return anchors


def baseline_floor_frontier_anchor_records(
    floor_anchors: list[BranchReplayRecord],
    replay_records: list[BranchReplayRecord],
) -> list[BranchReplayRecord]:
    represented_targets_by_profile: dict[str, set[int]] = {}
    for branch in floor_anchors:
        _context, target, _predicted, profile = branch_replay_parts(branch)
        represented_targets_by_profile.setdefault(profile, set()).add(target)
    frontier: list[BranchReplayRecord] = []
    seen_profile_targets: set[tuple[str, int]] = set()
    for branch in replay_records:
        context, target, _predicted, profile = branch_replay_parts(branch)
        represented_targets = represented_targets_by_profile.get(profile)
        if not represented_targets:
            continue
        if target in represented_targets:
            continue
        profile_target = (profile, target)
        if profile_target in seen_profile_targets:
            continue
        seen_profile_targets.add(profile_target)
        frontier.append((context, target, target, profile))
    return frontier


def baseline_floor_objective_anchor_batch(
    anchors: list[BranchReplayRecord],
    rng: random.Random,
    batch_size: int,
) -> list[BranchReplayRecord]:
    if not anchors:
        return []
    anchors_by_profile_target: dict[tuple[str, int], list[BranchReplayRecord]] = {}
    for branch in anchors:
        _context, target, _predicted, profile = branch_replay_parts(branch)
        anchors_by_profile_target.setdefault((profile, target), []).append(branch)
    profile_targets = list(anchors_by_profile_target)
    rng.shuffle(profile_targets)
    selected: list[BranchReplayRecord] = []
    for profile_target in profile_targets[: max(1, batch_size)]:
        selected.append(rng.choice(anchors_by_profile_target[profile_target]))
    return selected
