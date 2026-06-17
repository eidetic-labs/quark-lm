"""Specialized baseline-floor recovery outcome assessment."""

from __future__ import annotations

from typing import Any

from branch_diversity_snapshots import (
    branch_diversity_profile_delta_has_coverage_gain,
)
from transformer_baseline_floor_profile_outcome_types import (
    BaselineFloorBranchDiversityRecoveryOutcome,
    BaselineFloorCollapsedProfileBindingOutcome,
    BaselineFloorCoverageRecoveryOutcome,
    BaselineFloorMissingFirstTokenOutcome,
)


def evaluate_baseline_floor_coverage_recovery_outcome(
    *,
    recovery_floor_preserved: bool,
    recovery_score: tuple[float, ...],
    profile_base_score: tuple[float, ...] | None,
    recovery_delta: dict[str, Any],
    branch_stable_active: bool,
    prepared_score: tuple[float, ...] | None,
) -> BaselineFloorCoverageRecoveryOutcome:
    if not recovery_floor_preserved:
        return BaselineFloorCoverageRecoveryOutcome(
            accepted=False,
            outcome="floor_regressed",
            rejection_reason="floor_regression",
        )
    if profile_base_score is not None and recovery_score < profile_base_score:
        return BaselineFloorCoverageRecoveryOutcome(
            accepted=False,
            outcome="score_regressed",
            rejection_reason="score_regression",
        )
    if int(recovery_delta["regressed_profile_count"]) > 0:
        return BaselineFloorCoverageRecoveryOutcome(
            accepted=False,
            outcome="coverage_regressed",
            rejection_reason="coverage_regression",
        )
    if (
        branch_stable_active
        and prepared_score is not None
        and recovery_score < prepared_score
    ):
        return BaselineFloorCoverageRecoveryOutcome(
            accepted=False,
            outcome="branch_score_regressed",
            rejection_reason="branch_score_regression",
            branch_stability_preserved=False,
        )
    if int(recovery_delta["improved_profile_count"]) > 0:
        return BaselineFloorCoverageRecoveryOutcome(
            accepted=True,
            outcome="gained",
            rejection_reason="",
            branch_stability_preserved=True if branch_stable_active else None,
            branch_stable_accepted=branch_stable_active,
        )
    return BaselineFloorCoverageRecoveryOutcome(
        accepted=False,
        outcome="coverage_tied",
        rejection_reason="coverage_tie",
    )


def evaluate_baseline_floor_branch_diversity_recovery_outcome(
    *,
    floor_preserved: bool,
    recovery_score: tuple[float, ...],
    base_score: tuple[float, ...],
    coverage_delta: dict[str, Any],
) -> BaselineFloorBranchDiversityRecoveryOutcome:
    if not floor_preserved:
        return BaselineFloorBranchDiversityRecoveryOutcome(
            accepted=False,
            outcome="floor_regressed",
            rejection_reason="floor_regression",
        )
    if int(coverage_delta["regressed_profile_count"]) > 0:
        return BaselineFloorBranchDiversityRecoveryOutcome(
            accepted=False,
            outcome="coverage_regressed",
            rejection_reason="coverage_regression",
        )
    if recovery_score > base_score:
        return BaselineFloorBranchDiversityRecoveryOutcome(
            accepted=True,
            outcome="branch_diversity_improved",
            rejection_reason="",
        )
    if recovery_score == base_score:
        return BaselineFloorBranchDiversityRecoveryOutcome(
            accepted=False,
            outcome="score_tied",
            rejection_reason="score_tie",
        )
    return BaselineFloorBranchDiversityRecoveryOutcome(
        accepted=False,
        outcome="score_regressed",
        rejection_reason="score_regression",
    )


def evaluate_baseline_floor_collapsed_profile_binding_outcome(
    *,
    floor_preserved: bool,
    binding_score: tuple[float, ...],
    base_score: tuple[float, ...] | None,
    coverage_delta: dict[str, Any],
    profile_delta: dict[str, Any],
    owner_paraphrase_preservation_regressed: bool,
) -> BaselineFloorCollapsedProfileBindingOutcome:
    if not floor_preserved:
        return BaselineFloorCollapsedProfileBindingOutcome(
            accepted=False,
            outcome="floor_regressed",
            rejection_reason="floor_regression",
        )
    if int(coverage_delta["regressed_profile_count"]) > 0:
        return BaselineFloorCollapsedProfileBindingOutcome(
            accepted=False,
            outcome="coverage_regressed",
            rejection_reason="coverage_regression",
        )
    if int(profile_delta["regressed_profile_count"]) > 0:
        return BaselineFloorCollapsedProfileBindingOutcome(
            accepted=False,
            outcome="profile_diversity_regressed",
            rejection_reason="profile_diversity_regression",
        )
    if owner_paraphrase_preservation_regressed:
        return BaselineFloorCollapsedProfileBindingOutcome(
            accepted=False,
            outcome="preserved_profile_regressed",
            rejection_reason="owner_paraphrase_preservation_regression",
            owner_paraphrase_preservation_failed=True,
        )
    if base_score is not None and binding_score < base_score:
        return BaselineFloorCollapsedProfileBindingOutcome(
            accepted=False,
            outcome="score_regressed",
            rejection_reason="score_regression",
        )
    if int(profile_delta["improved_profile_count"]) > 0:
        return BaselineFloorCollapsedProfileBindingOutcome(
            accepted=True,
            outcome="collapsed_profile_improved",
            rejection_reason="",
        )
    return BaselineFloorCollapsedProfileBindingOutcome(
        accepted=False,
        outcome="collapsed_profile_tied",
        rejection_reason="collapsed_profile_tie",
    )


def evaluate_baseline_floor_missing_first_token_outcome(
    *,
    floor_preserved: bool,
    token_score: tuple[float, ...],
    base_score: tuple[float, ...] | None,
    coverage_delta: dict[str, Any],
    profile_delta: dict[str, Any],
) -> BaselineFloorMissingFirstTokenOutcome:
    if not floor_preserved:
        return BaselineFloorMissingFirstTokenOutcome(
            accepted=False,
            outcome="floor_regressed",
            rejection_reason="floor_regression",
        )
    if int(coverage_delta["regressed_profile_count"]) > 0:
        return BaselineFloorMissingFirstTokenOutcome(
            accepted=False,
            outcome="coverage_regressed",
            rejection_reason="coverage_regression",
        )
    if int(profile_delta["regressed_profile_count"]) > 0:
        return BaselineFloorMissingFirstTokenOutcome(
            accepted=False,
            outcome="target_profile_regressed",
            rejection_reason="target_profile_regression",
        )
    if base_score is not None and token_score < base_score:
        return BaselineFloorMissingFirstTokenOutcome(
            accepted=False,
            outcome="score_regressed",
            rejection_reason="score_regression",
        )
    if branch_diversity_profile_delta_has_coverage_gain(profile_delta):
        return BaselineFloorMissingFirstTokenOutcome(
            accepted=True,
            outcome="missing_first_token_coverage_gained",
            rejection_reason="",
        )
    return BaselineFloorMissingFirstTokenOutcome(
        accepted=False,
        outcome="missing_first_token_tied",
        rejection_reason="missing_first_token_tie",
    )

