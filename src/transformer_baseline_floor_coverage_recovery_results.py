"""Result records for baseline-floor coverage recovery attempts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CoverageRecoveryResult:
    loss_total: float
    loss_count: int
    floor_preserved: bool
    profile_probe_snapshot: dict[str, Any] | None
    profile_score: tuple[float, ...] | None
    diversity_outcome: str
    diversity_rejection_reason: str
    coverage_delta: dict[str, Any] | None
    coverage_outcome: str
    coverage_rejection_reason: str
    coverage_prep_accepted: bool
    attempted: bool = False
    accepted: bool = False
    outcome: str = "not_attempted"
    rejection_reason: str = ""
    learning_rate_scale: float | None = None
    records: int = 0
    delta: dict[str, Any] | None = None
    prepared_score: tuple[float, ...] | None = None
    score: tuple[float, ...] | None = None
    branch_stable_checked: bool = False
    branch_stable_accepted: bool = False
    branch_stability_preserved: bool | None = None


def initial_coverage_recovery_result(
    *,
    floor_preserved: bool,
    profile_probe_snapshot: dict[str, Any] | None,
    profile_score: tuple[float, ...] | None,
    diversity_outcome: str,
    diversity_rejection_reason: str,
    coverage_delta: dict[str, Any] | None,
    coverage_outcome: str,
    coverage_rejection_reason: str,
    coverage_prep_accepted: bool,
    prepared_score: tuple[float, ...] | None,
) -> CoverageRecoveryResult:
    return CoverageRecoveryResult(
        loss_total=0.0,
        loss_count=0,
        floor_preserved=floor_preserved,
        profile_probe_snapshot=profile_probe_snapshot,
        profile_score=profile_score,
        diversity_outcome=diversity_outcome,
        diversity_rejection_reason=diversity_rejection_reason,
        coverage_delta=coverage_delta,
        coverage_outcome=coverage_outcome,
        coverage_rejection_reason=coverage_rejection_reason,
        coverage_prep_accepted=coverage_prep_accepted,
        prepared_score=prepared_score,
    )


def accepted_coverage_recovery_result(
    *,
    loss_total: float,
    loss_count: int,
    recovery_floor_preserved: bool,
    probe_snapshot: dict[str, Any],
    recovery_score: tuple[float, ...],
    diversity_outcome: str,
    profile_base_score: tuple[float, ...] | None,
    recovery_delta: dict[str, Any],
    recovery_outcome: str,
    rejection_reason: str,
    recovery_learning_rate_scale: float,
    records: int,
    prepared_score: tuple[float, ...] | None,
    branch_stable_checked: bool,
    branch_stable_accepted: bool,
    branch_stability_preserved: bool | None,
) -> CoverageRecoveryResult:
    accepted_diversity_outcome = diversity_outcome
    if profile_base_score is not None:
        if recovery_score > profile_base_score:
            accepted_diversity_outcome = "improved"
        else:
            accepted_diversity_outcome = "tied"
    return CoverageRecoveryResult(
        loss_total=loss_total,
        loss_count=loss_count,
        floor_preserved=recovery_floor_preserved,
        profile_probe_snapshot=probe_snapshot,
        profile_score=recovery_score,
        diversity_outcome=accepted_diversity_outcome,
        diversity_rejection_reason="",
        coverage_delta=recovery_delta,
        coverage_outcome="gained",
        coverage_rejection_reason="",
        coverage_prep_accepted=False,
        attempted=True,
        accepted=True,
        outcome=recovery_outcome,
        rejection_reason=rejection_reason,
        learning_rate_scale=recovery_learning_rate_scale,
        records=records,
        delta=recovery_delta,
        prepared_score=prepared_score,
        score=recovery_score,
        branch_stable_checked=branch_stable_checked,
        branch_stable_accepted=branch_stable_accepted,
        branch_stability_preserved=branch_stability_preserved,
    )


def rejected_coverage_recovery_result(
    *,
    loss_total: float,
    loss_count: int,
    floor_preserved: bool,
    profile_probe_snapshot: dict[str, Any] | None,
    profile_score: tuple[float, ...] | None,
    diversity_outcome: str,
    diversity_rejection_reason: str,
    coverage_delta: dict[str, Any] | None,
    coverage_outcome: str,
    coverage_rejection_reason: str,
    coverage_prep_accepted: bool,
    attempted: bool,
    recovery_outcome: str,
    rejection_reason: str,
    records: int,
    recovery_delta: dict[str, Any] | None,
    prepared_score: tuple[float, ...] | None,
    recovery_score: tuple[float, ...] | None,
    branch_stable_checked: bool,
    branch_stable_accepted: bool,
    branch_stability_preserved: bool | None,
) -> CoverageRecoveryResult:
    return CoverageRecoveryResult(
        loss_total=loss_total,
        loss_count=loss_count,
        floor_preserved=floor_preserved,
        profile_probe_snapshot=profile_probe_snapshot,
        profile_score=profile_score,
        diversity_outcome=diversity_outcome,
        diversity_rejection_reason=diversity_rejection_reason,
        coverage_delta=coverage_delta,
        coverage_outcome=coverage_outcome,
        coverage_rejection_reason=coverage_rejection_reason,
        coverage_prep_accepted=coverage_prep_accepted,
        attempted=attempted,
        accepted=False,
        outcome=recovery_outcome,
        rejection_reason=rejection_reason,
        records=records,
        delta=recovery_delta,
        prepared_score=prepared_score,
        score=recovery_score,
        branch_stable_checked=branch_stable_checked,
        branch_stable_accepted=branch_stable_accepted,
        branch_stability_preserved=branch_stability_preserved,
    )
