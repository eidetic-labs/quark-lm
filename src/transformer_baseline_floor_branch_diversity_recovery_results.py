"""Branch-diversity recovery result builders."""

from __future__ import annotations

from typing import Any

from transformer_baseline_floor_branch_diversity_recovery_types import (
    BranchDiversityRecoveryResult,
)


def initial_branch_diversity_recovery_result(
    *,
    floor_preserved: bool,
    profile_probe_snapshot: dict[str, Any] | None,
    profile_score: tuple[float, ...] | None,
    diversity_outcome: str,
    diversity_rejection_reason: str,
) -> BranchDiversityRecoveryResult:
    return BranchDiversityRecoveryResult(
        loss_total=0.0,
        loss_count=0,
        floor_preserved=floor_preserved,
        profile_probe_snapshot=profile_probe_snapshot,
        profile_score=profile_score,
        diversity_outcome=diversity_outcome,
        diversity_rejection_reason=diversity_rejection_reason,
    )


def accepted_branch_diversity_recovery_result(
    *,
    loss_total: float,
    loss_count: int,
    floor_preserved: bool,
    profile_probe_snapshot: dict[str, Any],
    profile_score: tuple[float, ...],
    attempted: bool,
    outcome: str,
    rejection_reason: str,
    recovery_learning_rate_scale: float,
    records: int,
    base_score: tuple[float, ...],
    recovery_delta: dict[str, Any],
) -> BranchDiversityRecoveryResult:
    return BranchDiversityRecoveryResult(
        loss_total=loss_total,
        loss_count=loss_count,
        floor_preserved=floor_preserved,
        profile_probe_snapshot=profile_probe_snapshot,
        profile_score=profile_score,
        diversity_outcome="improved",
        diversity_rejection_reason="",
        attempted=attempted,
        accepted=True,
        outcome=outcome,
        rejection_reason=rejection_reason,
        learning_rate_scale=recovery_learning_rate_scale,
        records=records,
        base_score=base_score,
        score=profile_score,
        delta=recovery_delta,
    )


def fallback_branch_diversity_recovery_result(
    *,
    loss_total: float,
    loss_count: int,
    floor_preserved: bool,
    profile_probe_snapshot: dict[str, Any],
    profile_score: tuple[float, ...],
    diversity_outcome: str,
    diversity_rejection_reason: str,
    attempted: bool,
    outcome: str,
    rejection_reason: str,
    records: int,
    base_score: tuple[float, ...],
    recovery_score: tuple[float, ...] | None,
    recovery_delta: dict[str, Any] | None,
) -> BranchDiversityRecoveryResult:
    return BranchDiversityRecoveryResult(
        loss_total=loss_total,
        loss_count=loss_count,
        floor_preserved=floor_preserved,
        profile_probe_snapshot=profile_probe_snapshot,
        profile_score=profile_score,
        diversity_outcome=diversity_outcome,
        diversity_rejection_reason=diversity_rejection_reason,
        attempted=attempted,
        accepted=False,
        outcome=outcome,
        rejection_reason=rejection_reason,
        records=records,
        base_score=base_score,
        score=recovery_score,
        delta=recovery_delta,
    )
