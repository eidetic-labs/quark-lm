"""Baseline-floor anchor selection and profile-scale setup helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence

from replay_plan import BranchReplayRecord, branch_replay_parts
from transformer_baseline_floor_anchor_batches import (
    baseline_floor_frontier_anchor_records,
    baseline_floor_objective_anchor_batch,
    baseline_floor_repair_anchor_records,
)
from transformer_baseline_floor_anchor_profiles import (
    baseline_floor_anchor_profile_counts,
    baseline_floor_anchor_profile_groups,
    baseline_floor_anchor_profile_target_count,
)
from transformer_remaining_profile_binding import (
    remaining_profile_binding_profile_order,
    remaining_profile_binding_source_labels,
    source_profile_label,
)


@dataclass(frozen=True)
class BaselineFloorProfileSetup:
    """Prepared profile-scale state for baseline-floor stabilization."""

    profile_anchor_pool: list[BranchReplayRecord]
    profile_groups: dict[str, list[BranchReplayRecord]]
    frontier_targets_by_profile: dict[str, set[int]]
    profile_items: list[tuple[str, list[BranchReplayRecord]]]
    remaining_source_labels: set[str]
    remaining_source_profiles: list[str]


def baseline_floor_profile_setup(
    repair_anchors: list[BranchReplayRecord],
    frontier_anchors: list[BranchReplayRecord],
    remaining_binding_target_profiles: Sequence[str],
    include_frontier_anchors: bool,
    prioritize_remaining_profile_binding: bool,
) -> BaselineFloorProfileSetup:
    profile_anchor_pool = repair_anchors
    if include_frontier_anchors:
        profile_anchor_pool = repair_anchors + frontier_anchors
    profile_groups = baseline_floor_anchor_profile_groups(profile_anchor_pool)
    frontier_targets_by_profile = _frontier_targets_by_profile(
        frontier_anchors,
        include_frontier_anchors,
    )
    profile_items = list(profile_groups.items())
    remaining_source_labels: set[str] = set()
    remaining_source_profiles: list[str] = []
    if prioritize_remaining_profile_binding:
        target_profiles = list(remaining_binding_target_profiles)
        profile_items = remaining_profile_binding_profile_order(
            profile_groups,
            target_profiles,
        )
        remaining_source_labels = set(
            remaining_profile_binding_source_labels(target_profiles)
        )
        remaining_source_profiles = _remaining_source_profiles(
            profile_items,
            remaining_source_labels,
        )
    return BaselineFloorProfileSetup(
        profile_anchor_pool=profile_anchor_pool,
        profile_groups=profile_groups,
        frontier_targets_by_profile=frontier_targets_by_profile,
        profile_items=profile_items,
        remaining_source_labels=remaining_source_labels,
        remaining_source_profiles=remaining_source_profiles,
    )


def baseline_floor_profile_attempt(
    profile: str,
    profile_anchors: list[BranchReplayRecord],
    rng: random.Random,
    frontier_targets_by_profile: dict[str, set[int]],
    update_guard: dict[str, Any],
    *,
    frontier_active: bool,
    coverage_frontier_active: bool,
    coverage_prep_frontier_active: bool,
    diversity_active: bool,
    remaining_profile_binding_prioritized: bool,
    owner_paraphrase_binding_prioritized: bool,
    memory_consolidation_prioritized: bool,
) -> tuple[list[BranchReplayRecord], int]:
    profile_batch = baseline_floor_objective_anchor_batch(
        profile_anchors,
        rng,
        len(profile_anchors),
    )
    profile_frontier_records = 0
    if frontier_active:
        frontier_targets = frontier_targets_by_profile.get(profile, set())
        profile_frontier_records = sum(
            1
            for branch in profile_batch
            if branch_replay_parts(branch)[1] in frontier_targets
        )
    _record_profile_attempt_counts(
        update_guard,
        profile_batch,
        profile_frontier_records,
        frontier_active=frontier_active,
        coverage_frontier_active=coverage_frontier_active,
        coverage_prep_frontier_active=coverage_prep_frontier_active,
        diversity_active=diversity_active,
        remaining_profile_binding_prioritized=remaining_profile_binding_prioritized,
        owner_paraphrase_binding_prioritized=owner_paraphrase_binding_prioritized,
        memory_consolidation_prioritized=memory_consolidation_prioritized,
    )
    return profile_batch, profile_frontier_records


def _frontier_targets_by_profile(
    frontier_anchors: list[BranchReplayRecord],
    include_frontier_anchors: bool,
) -> dict[str, set[int]]:
    frontier_targets: dict[str, set[int]] = {}
    if not include_frontier_anchors:
        return frontier_targets
    for branch in frontier_anchors:
        _context, target, _predicted, frontier_profile = branch_replay_parts(branch)
        frontier_targets.setdefault(frontier_profile, set()).add(target)
    return frontier_targets


def _remaining_source_profiles(
    profile_items: list[tuple[str, list[BranchReplayRecord]]],
    remaining_source_labels: set[str],
) -> list[str]:
    return [
        profile
        for profile, anchors in profile_items
        if source_profile_label(profile) in remaining_source_labels
        and _target_count(anchors) > 1
    ]


def _target_count(anchors: list[BranchReplayRecord]) -> int:
    return len(
        {
            target
            for _context, target, _predicted, _profile in (
                branch_replay_parts(anchor) for anchor in anchors
            )
        }
    )


def _record_profile_attempt_counts(
    update_guard: dict[str, Any],
    profile_batch: list[BranchReplayRecord],
    profile_frontier_records: int,
    *,
    frontier_active: bool,
    coverage_frontier_active: bool,
    coverage_prep_frontier_active: bool,
    diversity_active: bool,
    remaining_profile_binding_prioritized: bool,
    owner_paraphrase_binding_prioritized: bool,
    memory_consolidation_prioritized: bool,
) -> None:
    update_guard["sequential_profile_attempts"] += 1
    update_guard["profile_scale_memory_attempts"] += 1
    if frontier_active:
        update_guard["profile_scale_frontier_attempts"] += 1
        update_guard["profile_scale_frontier_records"] += profile_frontier_records
    if coverage_frontier_active:
        update_guard["profile_scale_coverage_frontier_attempts"] += 1
    if coverage_prep_frontier_active:
        update_guard["profile_scale_coverage_prep_frontier_attempts"] += 1
    if diversity_active:
        update_guard["profile_scale_diversity_attempts"] += 1
    if remaining_profile_binding_prioritized:
        update_guard[
            "profile_scale_remaining_profile_binding_prioritized_attempts"
        ] += 1
    if owner_paraphrase_binding_prioritized:
        update_guard[
            "profile_scale_owner_paraphrase_binding_prioritized_attempts"
        ] += 1
    if memory_consolidation_prioritized:
        update_guard[
            "profile_scale_memory_consolidation_prioritized_attempts"
        ] += 1
    update_guard["sequential_profile_records"] += len(profile_batch)
    update_guard["stabilization_anchor_batches"] += 1
    update_guard["stabilization_anchor_records"] += len(profile_batch)
