"""Baseline-floor profile-scale outcome assessment."""

from __future__ import annotations

from typing import Any, Callable

from branch_diversity_snapshots import (
    branch_diversity_snapshot_preserves_target_coverage,
    branch_diversity_snapshot_score,
    branch_diversity_snapshot_target_coverage_delta,
)
from transformer_baseline_floor_profile_outcome_types import (
    BaselineFloorBranchDiversityRecoveryOutcome,
    BaselineFloorCollapsedProfileBindingOutcome,
    BaselineFloorCoverageRecoveryOutcome,
    BaselineFloorMissingFirstTokenOutcome,
    BaselineFloorProfileOutcome,
)
from transformer_baseline_floor_recovery_outcomes import (
    evaluate_baseline_floor_branch_diversity_recovery_outcome,
    evaluate_baseline_floor_collapsed_profile_binding_outcome,
    evaluate_baseline_floor_coverage_recovery_outcome,
    evaluate_baseline_floor_missing_first_token_outcome,
)


def evaluate_baseline_floor_profile_outcome(
    *,
    profile_probe_snapshot: dict[str, Any],
    direct_baseline: dict[str, Any],
    profile_base_snapshot: dict[str, Any] | None,
    profile_base_score: tuple[float, ...] | None,
    diversity_active: bool,
    coverage_frontier_active: bool,
    coverage_prep_frontier_active: bool,
    preserves_target_coverage: Callable[
        [dict[str, Any], dict[str, Any]], bool
    ] = branch_diversity_snapshot_preserves_target_coverage,
    snapshot_score: Callable[[dict[str, Any]], tuple[float, ...]] = (
        branch_diversity_snapshot_score
    ),
    target_coverage_delta: Callable[
        [dict[str, Any], dict[str, Any]], dict[str, Any]
    ] = branch_diversity_snapshot_target_coverage_delta,
) -> BaselineFloorProfileOutcome:
    floor_preserved = preserves_target_coverage(profile_probe_snapshot, direct_baseline)
    diversity_outcome = "not_active"
    diversity_rejection_reason = "floor_regression"
    profile_score: tuple[float, ...] | None = None
    coverage_outcome = "not_active"
    coverage_rejection_reason = "floor_regression"
    coverage_delta: dict[str, Any] | None = None

    if diversity_active:
        profile_score = snapshot_score(profile_probe_snapshot)
        if floor_preserved and profile_base_score is not None:
            if profile_score > profile_base_score:
                diversity_outcome = "improved"
                diversity_rejection_reason = ""
            elif profile_score == profile_base_score:
                diversity_outcome = "tied"
                diversity_rejection_reason = ""
            else:
                diversity_outcome = "regressed"
                diversity_rejection_reason = "score_regression"
        else:
            diversity_outcome = "floor_regressed"

    if coverage_frontier_active and profile_base_snapshot is not None:
        coverage_delta = target_coverage_delta(
            profile_probe_snapshot,
            profile_base_snapshot,
        )
        if floor_preserved:
            if int(coverage_delta["regressed_profile_count"]) > 0:
                coverage_outcome = "regressed"
                coverage_rejection_reason = "coverage_regression"
            elif int(coverage_delta["improved_profile_count"]) > 0:
                coverage_outcome = "gained"
                coverage_rejection_reason = ""
            else:
                coverage_outcome = "tied"
                coverage_rejection_reason = "coverage_tie"
        else:
            coverage_outcome = "floor_regressed"

    coverage_prep_accepted = (
        coverage_prep_frontier_active
        and coverage_outcome == "tied"
        and diversity_outcome == "improved"
    )
    return BaselineFloorProfileOutcome(
        floor_preserved=floor_preserved,
        diversity_outcome=diversity_outcome,
        diversity_rejection_reason=diversity_rejection_reason,
        profile_score=profile_score,
        coverage_outcome=coverage_outcome,
        coverage_rejection_reason=coverage_rejection_reason,
        coverage_delta=coverage_delta,
        coverage_prep_accepted=coverage_prep_accepted,
    )
