"""Target selection for baseline-floor binding recovery stages."""

from __future__ import annotations

from typing import Any, Callable

from branch_diversity_snapshots import (
    branch_diversity_snapshot_collapsed_profile_names,
)


def select_collapsed_profile_binding_targets(
    profile_probe_snapshot: dict[str, Any],
    *,
    memory_consolidation_active: bool,
    memory_consolidation_target_profiles: list[str] | tuple[str, ...] | set[str],
    owner_paraphrase_binding_active: bool,
    owner_paraphrase_target_profiles: list[str] | tuple[str, ...] | set[str],
    collapsed_profile_names: Callable[[dict[str, Any]], list[str]] = (
        branch_diversity_snapshot_collapsed_profile_names
    ),
) -> list[str]:
    target_profiles = collapsed_profile_names(profile_probe_snapshot)
    if memory_consolidation_active:
        memory_targets = set(memory_consolidation_target_profiles)
        return [name for name in target_profiles if name in memory_targets]
    if owner_paraphrase_binding_active:
        owner_targets = set(owner_paraphrase_target_profiles)
        return [name for name in target_profiles if name in owner_targets]
    return target_profiles
