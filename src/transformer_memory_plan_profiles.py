"""Profile selection for memory-consolidation source plans."""

from __future__ import annotations

from typing import Any


def ordered_memory_consolidation_profiles(raw_profiles: Any) -> list[str]:
    profiles: list[str] = []
    seen: set[str] = set()
    if not isinstance(raw_profiles, list):
        return profiles
    for raw_profile in raw_profiles:
        if not isinstance(raw_profile, str) or raw_profile in seen:
            continue
        profiles.append(raw_profile)
        seen.add(raw_profile)
    return profiles


def memory_consolidation_source_plan_targets(
    source_plan: dict[str, Any],
    max_profiles: int,
    *,
    require_collapsed_targets: bool = False,
) -> tuple[dict[str, Any], list[str], list[str], list[str]]:
    if max_profiles < 1:
        raise ValueError("memory consolidation max profiles must be at least 1")
    if source_plan.get("kind") != "memory_consolidation_plan":
        raise ValueError("memory consolidation source plan must be a memory_consolidation_plan")
    raw_summary = source_plan.get("summary", {})
    if not isinstance(raw_summary, dict):
        raise ValueError("memory consolidation source plan summary must be an object")
    summary = dict(raw_summary)
    collapsed_profiles = ordered_memory_consolidation_profiles(
        summary.get("collapsed_memory_backed_profiles", [])
    )
    if require_collapsed_targets and not collapsed_profiles:
        raise ValueError(
            "remaining-collapsed memory consolidation mode requires "
            "collapsed_memory_backed_profiles in the source plan"
        )
    top_priority_profiles = ordered_memory_consolidation_profiles(
        summary.get("top_priority_profiles", [])
    )
    targets = collapsed_profiles or top_priority_profiles
    if not targets:
        priority_records = source_plan.get("profile_priorities", [])
        if isinstance(priority_records, list):
            targets = ordered_memory_consolidation_profiles(
                [
                    record.get("profile")
                    for record in priority_records
                    if isinstance(record, dict)
                ]
            )
    if not targets:
        raise ValueError("memory consolidation source plan has no target profiles")
    return (
        summary,
        targets[:max_profiles],
        top_priority_profiles,
        collapsed_profiles,
    )
