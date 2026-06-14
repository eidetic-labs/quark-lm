"""Replay planning primitives for QuarkLM branch repair screens."""

from __future__ import annotations

from typing import Any


BranchReplayRecord = tuple[list[int], int, int] | tuple[list[int], int, int, str]
ProfiledBranchSeed = tuple[list[int], int, str]


def branch_replay_parts(
    branch: BranchReplayRecord,
) -> tuple[list[int], int, int, str]:
    if len(branch) >= 4:
        return branch[0], branch[1], branch[2], branch[3]
    return branch[0], branch[1], branch[2], "__all__"


def profile_key_from_source(source: str | None) -> str:
    return source or "unknown"


def direct_answer_profile_key(example: Any) -> str:
    return profile_key_from_source(getattr(example, "source", None))


def branch_replay_profile_groups(
    branches: list[BranchReplayRecord],
    profile_aware_targets: bool = False,
) -> dict[str, list[tuple[list[int], int, int, str]]]:
    groups: dict[str, list[tuple[list[int], int, int, str]]] = {}
    for branch in branches:
        context, target, predicted, profile = branch_replay_parts(branch)
        profile_key = profile if profile_aware_targets else "__all__"
        groups.setdefault(profile_key, []).append(
            (context, target, predicted, profile)
        )
    return groups


def branch_replay_plan(
    branches: list[BranchReplayRecord],
    replay_branches: list[BranchReplayRecord],
    profile_aware_targets: bool = False,
) -> dict[str, Any]:
    branch_groups = branch_replay_profile_groups(branches, profile_aware_targets)
    replay_groups = branch_replay_profile_groups(
        replay_branches if replay_branches else branches,
        profile_aware_targets,
    )
    profiles = sorted(set(branch_groups) | set(replay_groups))
    profile_summaries: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        branch_parts = branch_groups.get(profile, [])
        replay_parts = replay_groups.get(profile, [])
        target_ids = sorted(
            {target for _context, target, _predicted, _profile in replay_parts}
        )
        represented_target_ids = sorted(
            {
                predicted
                for _context, _target, predicted, _profile in replay_parts
                if predicted in target_ids
            }
        )
        missing_target_ids = sorted(set(target_ids) - set(represented_target_ids))
        profile_summaries[profile] = {
            "branch_count": len(branch_parts),
            "replay_count": len(replay_parts),
            "target_ids": target_ids,
            "target_count": len(target_ids),
            "represented_target_ids": represented_target_ids,
            "represented_target_count": len(represented_target_ids),
            "missing_target_ids": missing_target_ids,
            "missing_target_count": len(missing_target_ids),
            "coverage_floor": (
                len(represented_target_ids) / len(target_ids)
                if target_ids
                else 1.0
            ),
        }
    return {
        "profile_aware_targets": profile_aware_targets,
        "branch_count": len(branches),
        "replay_count": len(replay_branches if replay_branches else branches),
        "profiles": profile_summaries,
    }
