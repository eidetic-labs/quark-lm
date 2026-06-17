"""Profile-scale baseline-floor attempt planning."""

from __future__ import annotations

from typing import Any

from transformer_baseline_floor_anchor_selection import baseline_floor_profile_attempt


def record_baseline_floor_profile_scale_remaining_sources(
    ctx: Any,
    remaining_sources: list[str],
) -> None:
    setup = ctx.direct_setup
    if not (
        setup.direct_answer_baseline_floor_profile_scale_remaining_profile_binding_frontier_stabilization_active
    ):
        return
    ctx.update_guard["profile_scale_remaining_profile_binding_source_profiles"] = (
        remaining_sources
    )
    if (
        setup.direct_answer_baseline_floor_profile_scale_owner_paraphrase_binding_frontier_stabilization_active
    ):
        ctx.update_guard["profile_scale_owner_paraphrase_binding_source_profiles"] = (
            remaining_sources
        )


def baseline_floor_profile_scale_priorities(
    setup: Any,
    profile: str,
    remaining_source_profiles: list[str],
) -> dict[str, bool]:
    prioritized = profile in remaining_source_profiles
    return {
        "remaining": (
            setup.direct_answer_baseline_floor_profile_scale_remaining_profile_binding_frontier_stabilization_active
            and prioritized
        ),
        "owner": (
            setup.direct_answer_baseline_floor_profile_scale_owner_paraphrase_binding_frontier_stabilization_active
            and prioritized
        ),
        "memory": (
            setup.direct_answer_baseline_floor_profile_scale_memory_consolidation_frontier_stabilization_active
            and prioritized
        ),
    }


def select_baseline_floor_profile_scale_batch(
    ctx: Any,
    *,
    profile: str,
    profile_anchors: list[Any],
    frontier_targets_by_profile: dict[str, set[int]],
    priorities: dict[str, bool],
) -> tuple[list[Any], int]:
    setup = ctx.direct_setup
    return baseline_floor_profile_attempt(
        profile=profile,
        profile_anchors=profile_anchors,
        rng=ctx.rng,
        frontier_targets_by_profile=frontier_targets_by_profile,
        update_guard=ctx.update_guard,
        frontier_active=(
            setup.direct_answer_baseline_floor_profile_scale_frontier_stabilization_active
        ),
        coverage_frontier_active=(
            setup.direct_answer_baseline_floor_profile_scale_coverage_frontier_stabilization_active
        ),
        coverage_prep_frontier_active=(
            setup.direct_answer_baseline_floor_profile_scale_coverage_prep_frontier_stabilization_active
        ),
        diversity_active=(
            setup.direct_answer_baseline_floor_profile_scale_diversity_stabilization_active
        ),
        remaining_profile_binding_prioritized=priorities["remaining"],
        owner_paraphrase_binding_prioritized=priorities["owner"],
        memory_consolidation_prioritized=priorities["memory"],
    )
