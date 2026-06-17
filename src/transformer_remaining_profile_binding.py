"""Profile routing helpers for remaining-profile binding."""

from __future__ import annotations

from replay_plan import BranchReplayRecord, branch_replay_parts
from transformer_direct_modes import (
    BASELINE_FLOOR_REMAINING_PROFILE_BINDING_TARGET_SOURCE_LABELS,
)


def source_profile_label(profile: str) -> str:
    if ":" not in profile:
        return profile
    return profile.split(":", 1)[1]


def remaining_profile_binding_source_labels(
    target_profiles: list[str] | set[str] | tuple[str, ...],
) -> list[str]:
    labels: set[str] = set()
    for target_profile in target_profiles:
        mapped_labels = BASELINE_FLOOR_REMAINING_PROFILE_BINDING_TARGET_SOURCE_LABELS.get(
            target_profile
        )
        if mapped_labels is None:
            labels.add(target_profile)
        else:
            labels.update(mapped_labels)
    return sorted(labels)


def remaining_profile_binding_profile_order(
    profile_groups: dict[str, list[BranchReplayRecord]],
    target_profiles: list[str] | set[str] | tuple[str, ...],
) -> list[tuple[str, list[BranchReplayRecord]]]:
    source_labels = set(remaining_profile_binding_source_labels(target_profiles))

    def priority(item: tuple[str, list[BranchReplayRecord]]) -> tuple[int, str, str]:
        profile, anchors = item
        label = source_profile_label(profile)
        target_count = len(
            {
                target
                for _context, target, _predicted, _profile in (
                    branch_replay_parts(anchor) for anchor in anchors
                )
            }
        )
        priority_rank = 0 if label in source_labels and target_count > 1 else 1
        return priority_rank, label, profile

    return sorted(profile_groups.items(), key=priority)
