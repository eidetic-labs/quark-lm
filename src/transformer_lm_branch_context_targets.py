"""Replay target bookkeeping for branch-context objectives."""

from __future__ import annotations

from dataclasses import dataclass

from replay_plan import BranchReplayRecord, branch_replay_parts


BranchReplayPart = tuple[list[int], int, int, str]


@dataclass(frozen=True)
class BranchContextReplayTargetState:
    branch_parts: list[BranchReplayPart]
    replay_parts: list[BranchReplayPart]
    floor_preservation_parts: list[BranchReplayPart]
    replay_record_count: int
    replay_targets: list[int]
    replay_target_set: set[int]
    replay_target_offsets: dict[int, int]
    replay_targets_by_profile: dict[str, list[int]]
    replay_target_sets_by_profile: dict[str, set[int]]
    replay_target_offsets_by_profile: dict[str, dict[int, int]]
    deficit_targets_by_profile: dict[str, set[int]]


def build_branch_context_replay_target_state(
    branches: list[BranchReplayRecord],
    replay_branches: list[BranchReplayRecord],
    floor_preservation_branches: list[BranchReplayRecord] | None,
    *,
    profile_aware_targets: bool,
) -> BranchContextReplayTargetState:
    replay_parts = [branch_replay_parts(branch) for branch in replay_branches]
    branch_parts = [branch_replay_parts(branch) for branch in branches]
    floor_preservation_parts = [
        branch_replay_parts(branch) for branch in (floor_preservation_branches or [])
    ]
    replay_targets = sorted(
        {target for _context, target, _predicted, _profile in replay_parts}
    )
    if not replay_targets:
        replay_targets = sorted(
            {target for _context, target, _predicted, _profile in branch_parts}
        )
        replay_parts = branch_parts

    replay_target_set = set(replay_targets)
    replay_target_offsets = {target: offset for offset, target in enumerate(replay_targets)}
    replay_target_sets_by_profile = _target_sets_by_profile(
        replay_parts,
        replay_target_set,
        replay_targets,
        profile_aware_targets=profile_aware_targets,
    )
    replay_targets_by_profile, replay_target_offsets_by_profile = _profile_offsets(
        replay_target_sets_by_profile
    )
    deficit_targets_by_profile = _deficit_targets_by_profile(
        replay_parts,
        replay_target_sets_by_profile,
        replay_target_set,
        profile_aware_targets=profile_aware_targets,
    )
    return BranchContextReplayTargetState(
        branch_parts=branch_parts,
        replay_parts=replay_parts,
        floor_preservation_parts=floor_preservation_parts,
        replay_record_count=len(replay_parts),
        replay_targets=replay_targets,
        replay_target_set=replay_target_set,
        replay_target_offsets=replay_target_offsets,
        replay_targets_by_profile=replay_targets_by_profile,
        replay_target_sets_by_profile=replay_target_sets_by_profile,
        replay_target_offsets_by_profile=replay_target_offsets_by_profile,
        deficit_targets_by_profile=deficit_targets_by_profile,
    )


def branch_context_profile_key(profile: str, profile_aware_targets: bool) -> str:
    return profile if profile_aware_targets else "__all__"


def _target_sets_by_profile(
    replay_parts: list[BranchReplayPart],
    replay_target_set: set[int],
    replay_targets: list[int],
    *,
    profile_aware_targets: bool,
) -> dict[str, set[int]]:
    target_sets: dict[str, set[int]] = {}
    for _context, target, _predicted, profile in replay_parts:
        profile_key = branch_context_profile_key(profile, profile_aware_targets)
        target_sets.setdefault(profile_key, set()).add(target)
    if not target_sets:
        target_sets["__all__"] = set(replay_targets)
    if not profile_aware_targets:
        target_sets["__all__"] = replay_target_set
    return target_sets


def _profile_offsets(
    target_sets: dict[str, set[int]],
) -> tuple[dict[str, list[int]], dict[str, dict[int, int]]]:
    targets_by_profile: dict[str, list[int]] = {}
    offsets_by_profile: dict[str, dict[int, int]] = {}
    for profile_key, profile_targets in target_sets.items():
        ordered_targets = sorted(profile_targets)
        targets_by_profile[profile_key] = ordered_targets
        offsets_by_profile[profile_key] = {
            target: offset for offset, target in enumerate(ordered_targets)
        }
    return targets_by_profile, offsets_by_profile


def _deficit_targets_by_profile(
    replay_parts: list[BranchReplayPart],
    target_sets_by_profile: dict[str, set[int]],
    replay_target_set: set[int],
    *,
    profile_aware_targets: bool,
) -> dict[str, set[int]]:
    predicted_targets_by_profile: dict[str, set[int]] = {}
    for _context, _target, predicted, profile in replay_parts:
        profile_key = branch_context_profile_key(profile, profile_aware_targets)
        profile_target_set = target_sets_by_profile.get(profile_key, replay_target_set)
        if predicted in profile_target_set:
            predicted_targets_by_profile.setdefault(profile_key, set()).add(predicted)
    return {
        profile_key: profile_target_set
        - predicted_targets_by_profile.get(profile_key, set())
        for profile_key, profile_target_set in target_sets_by_profile.items()
    }
