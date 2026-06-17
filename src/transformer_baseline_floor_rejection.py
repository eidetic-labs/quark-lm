"""Baseline-floor profile-scale rejection accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from branch_diversity_snapshot_coverage import branch_diversity_snapshot_target_coverage_diagnostics
from transformer_baseline_floor_rejection_counts import (
    increment_rejection_counter,
    increment_rejection_reason,
)
from transformer_baseline_floor_rejection_samples import (
    record_baseline_floor_profile_rejection_sample,
)


@dataclass(frozen=True)
class BaselineFloorProfileRejectionAccounting:
    remaining_profile_binding_prioritized: bool = False
    owner_paraphrase_binding_prioritized: bool = False
    memory_consolidation_prioritized: bool = False
    diversity_active: bool = False
    floor_preserved: bool = False
    diversity_accepted: bool = False
    diversity_rejection_reason: str = ""
    frontier_active: bool = False
    coverage_frontier_active: bool = False
    coverage_outcome: str = "not_active"
    coverage_rejection_reason: str = ""
    coverage_prep_active: bool = False


def record_baseline_floor_profile_rejection(
    update_guard: dict[str, Any],
    accounting: BaselineFloorProfileRejectionAccounting,
) -> str:
    diversity_rejection_reason = accounting.diversity_rejection_reason
    increment_rejection_counter(update_guard, "sequential_profile_rejections")
    increment_rejection_counter(update_guard, "profile_scale_memory_rejections")
    if accounting.remaining_profile_binding_prioritized:
        increment_rejection_counter(
            update_guard,
            "profile_scale_remaining_profile_binding_prioritized_rejections",
        )
    if accounting.owner_paraphrase_binding_prioritized:
        increment_rejection_counter(
            update_guard,
            "profile_scale_owner_paraphrase_binding_prioritized_rejections",
        )
    if accounting.memory_consolidation_prioritized:
        increment_rejection_counter(
            update_guard,
            "profile_scale_memory_consolidation_prioritized_rejections",
        )
    if accounting.diversity_active:
        increment_rejection_counter(update_guard, "profile_scale_diversity_rejections")
        if not accounting.floor_preserved:
            increment_rejection_counter(
                update_guard,
                "profile_scale_diversity_floor_rejections",
            )
        elif not accounting.diversity_accepted:
            increment_rejection_counter(
                update_guard,
                "profile_scale_diversity_score_regressions",
            )
        else:
            diversity_rejection_reason = "coverage_frontier_rejection"
        increment_rejection_reason(
            update_guard,
            "profile_scale_diversity_rejection_reasons",
            diversity_rejection_reason,
        )
    if accounting.frontier_active:
        increment_rejection_counter(update_guard, "profile_scale_frontier_rejections")
    if accounting.coverage_frontier_active:
        increment_rejection_counter(
            update_guard,
            "profile_scale_coverage_frontier_rejections",
        )
        if accounting.coverage_outcome == "gained":
            increment_rejection_counter(
                update_guard,
                "profile_scale_coverage_frontier_gains",
            )
        elif accounting.coverage_outcome == "tied":
            increment_rejection_counter(
                update_guard,
                "profile_scale_coverage_frontier_ties",
            )
        elif accounting.coverage_outcome in {"regressed", "floor_regressed"}:
            increment_rejection_counter(
                update_guard,
                "profile_scale_coverage_frontier_regressions",
            )
        increment_rejection_reason(
            update_guard,
            "profile_scale_coverage_frontier_rejection_reasons",
            accounting.coverage_rejection_reason or "not_accepted",
        )
    if accounting.coverage_prep_active:
        increment_rejection_counter(
            update_guard,
            "profile_scale_coverage_prep_frontier_rejections",
        )
        reason = (
            accounting.coverage_rejection_reason
            or diversity_rejection_reason
            or "not_accepted"
        )
        increment_rejection_reason(
            update_guard,
            "profile_scale_coverage_prep_frontier_rejection_reasons",
            reason,
        )
    return diversity_rejection_reason


def record_baseline_floor_profile_attempt_rejection(
    update_guard: dict[str, Any],
    *,
    profile: str,
    records: int,
    frontier_records: int,
    learning_rate_scale: float,
    scale_key: str,
    direct_baseline: dict[str, Any],
    profile_probe_snapshot: dict[str, Any],
    remaining_profile_binding_prioritized: bool,
    owner_paraphrase_binding_prioritized: bool,
    memory_consolidation_prioritized: bool,
    diversity_active: bool,
    floor_preserved: bool,
    diversity_accepted: bool,
    diversity_outcome: str,
    diversity_rejection_reason: str,
    profile_score: tuple[float, ...] | None,
    profile_base_score: tuple[float, ...] | None,
    frontier_active: bool,
    coverage_frontier_active: bool,
    coverage_delta: dict[str, Any] | None,
    coverage_outcome: str,
    coverage_prep_active: bool,
    coverage_prep_accepted: bool,
    coverage_rejection_reason: str,
    target_coverage_diagnostics: Callable[
        [dict[str, Any], dict[str, Any]], dict[str, Any]
    ] = branch_diversity_snapshot_target_coverage_diagnostics,
) -> str:
    rejection_reason = record_baseline_floor_profile_rejection(
        update_guard,
        BaselineFloorProfileRejectionAccounting(
            remaining_profile_binding_prioritized=(
                remaining_profile_binding_prioritized
            ),
            owner_paraphrase_binding_prioritized=(
                owner_paraphrase_binding_prioritized
            ),
            memory_consolidation_prioritized=memory_consolidation_prioritized,
            diversity_active=diversity_active,
            floor_preserved=floor_preserved,
            diversity_accepted=diversity_accepted,
            diversity_rejection_reason=diversity_rejection_reason,
            frontier_active=frontier_active,
            coverage_frontier_active=coverage_frontier_active,
            coverage_outcome=coverage_outcome,
            coverage_rejection_reason=coverage_rejection_reason,
            coverage_prep_active=coverage_prep_active,
        ),
    )
    diagnostics = target_coverage_diagnostics(
        profile_probe_snapshot,
        direct_baseline,
    )
    record_baseline_floor_profile_rejection_sample(
        update_guard,
        profile=profile,
        records=records,
        frontier_records=frontier_records,
        learning_rate_scale=learning_rate_scale,
        scale_key=scale_key,
        diagnostics=diagnostics,
        diversity_active=diversity_active,
        profile_score=profile_score,
        profile_base_score=profile_base_score,
        diversity_outcome=diversity_outcome,
        diversity_rejection_reason=rejection_reason,
        coverage_active=coverage_frontier_active,
        coverage_delta=coverage_delta,
        coverage_outcome=coverage_outcome,
        coverage_prep_accepted=coverage_prep_accepted,
        coverage_rejection_reason=coverage_rejection_reason,
    )
    return rejection_reason
