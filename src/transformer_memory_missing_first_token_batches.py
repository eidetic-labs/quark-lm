"""Anchor batch selection for missing-first-token memory consolidation."""

from __future__ import annotations

import random

from replay_plan import BranchReplayRecord, branch_replay_parts


def missing_first_token_anchor_batch(
    anchors: list[BranchReplayRecord],
    target_ids: set[int],
    rng: random.Random,
    batch_size: int,
) -> list[BranchReplayRecord]:
    if not anchors or not target_ids:
        return []
    by_target: dict[int, list[BranchReplayRecord]] = {}
    for branch in anchors:
        _context, target, _predicted, _profile = branch_replay_parts(branch)
        if target in target_ids:
            by_target.setdefault(target, []).append(branch)
    if not by_target:
        return []
    targets = list(by_target)
    rng.shuffle(targets)
    selected: list[BranchReplayRecord] = []
    for target in targets[: max(1, batch_size)]:
        selected.append(rng.choice(by_target[target]))
    return selected
