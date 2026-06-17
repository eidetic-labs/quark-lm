"""Result builders for baseline-floor memory consolidation attempts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from transformer_memory_plan_helpers import MissingFirstTokenTargetPlan


@dataclass(frozen=True)
class MissingFirstTokenResult:
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
    attempted: bool = False
    accepted: bool = False
    outcome: str = "not_attempted"
    rejection_reason: str = ""
    learning_rate_scale: float | None = None
    records: int = 0
    target_profiles: list[str] | None = None
    target_ids: list[int] | None = None
    base_score: tuple[float, ...] | None = None
    score: tuple[float, ...] | None = None
    delta: dict[str, Any] | None = None


def initial_missing_first_token_result(
    *,
    floor_preserved: bool,
    profile_probe_snapshot: dict[str, Any] | None,
    profile_score: tuple[float, ...] | None,
    diversity_outcome: str,
    diversity_rejection_reason: str,
    coverage_delta: dict[str, Any] | None,
    coverage_outcome: str,
    coverage_rejection_reason: str,
) -> MissingFirstTokenResult:
    return MissingFirstTokenResult(
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
        target_profiles=[],
        target_ids=[],
    )


def missing_first_token_target_result(
    base_result: MissingFirstTokenResult,
    target_plan: MissingFirstTokenTargetPlan,
) -> MissingFirstTokenResult:
    return MissingFirstTokenResult(
        loss_total=base_result.loss_total,
        loss_count=base_result.loss_count,
        floor_preserved=base_result.floor_preserved,
        profile_probe_snapshot=base_result.profile_probe_snapshot,
        profile_score=base_result.profile_score,
        diversity_outcome=base_result.diversity_outcome,
        diversity_rejection_reason=base_result.diversity_rejection_reason,
        coverage_delta=base_result.coverage_delta,
        coverage_outcome=base_result.coverage_outcome,
        coverage_rejection_reason=base_result.coverage_rejection_reason,
        target_profiles=target_plan.target_profiles,
        target_ids=target_plan.target_ids,
    )


def accepted_missing_first_token_result(
    *,
    loss_total: float,
    loss_count: int,
    floor_preserved: bool,
    profile_probe_snapshot: dict[str, Any],
    profile_score: tuple[float, ...],
    diversity_outcome: str,
    coverage_delta: dict[str, Any] | None,
    attempted: bool,
    outcome: str,
    rejection_reason: str,
    learning_rate_scale: float,
    records: int,
    target_plan: MissingFirstTokenTargetPlan,
    base_score: tuple[float, ...] | None,
    delta: dict[str, Any] | None,
) -> MissingFirstTokenResult:
    return MissingFirstTokenResult(
        loss_total=loss_total,
        loss_count=loss_count,
        floor_preserved=floor_preserved,
        profile_probe_snapshot=profile_probe_snapshot,
        profile_score=profile_score,
        diversity_outcome=diversity_outcome,
        diversity_rejection_reason="",
        coverage_delta=coverage_delta,
        coverage_outcome="gained",
        coverage_rejection_reason="",
        attempted=attempted,
        accepted=True,
        outcome=outcome,
        rejection_reason=rejection_reason,
        learning_rate_scale=learning_rate_scale,
        records=records,
        target_profiles=target_plan.target_profiles,
        target_ids=target_plan.target_ids,
        base_score=base_score,
        score=profile_score,
        delta=delta,
    )


def fallback_missing_first_token_result(
    *,
    base_result: MissingFirstTokenResult,
    loss_total: float,
    loss_count: int,
    attempted: bool,
    outcome: str,
    rejection_reason: str,
    records: int,
    target_plan: MissingFirstTokenTargetPlan,
    base_score: tuple[float, ...] | None,
    score: tuple[float, ...] | None,
    delta: dict[str, Any] | None,
) -> MissingFirstTokenResult:
    return MissingFirstTokenResult(
        loss_total=loss_total,
        loss_count=loss_count,
        floor_preserved=base_result.floor_preserved,
        profile_probe_snapshot=base_result.profile_probe_snapshot,
        profile_score=base_result.profile_score,
        diversity_outcome=base_result.diversity_outcome,
        diversity_rejection_reason=base_result.diversity_rejection_reason,
        coverage_delta=base_result.coverage_delta,
        coverage_outcome=base_result.coverage_outcome,
        coverage_rejection_reason=base_result.coverage_rejection_reason,
        attempted=attempted,
        accepted=False,
        outcome=outcome,
        rejection_reason=rejection_reason,
        records=records,
        target_profiles=target_plan.target_profiles,
        target_ids=target_plan.target_ids,
        base_score=base_score,
        score=score,
        delta=delta,
    )


def missing_first_token_probe_metadata(
    *,
    profile_scale: float,
    missing_token_scale: float,
    update_shape: str,
    profile: str,
    profile_batch_size: int,
    profile_frontier_records: int,
    records: int,
    target_plan: MissingFirstTokenTargetPlan,
    profile_specific: bool,
    target_profiles: Sequence[str],
) -> dict[str, Any]:
    return {
        "baseline_floor_update_guard_probe": True,
        "baseline_floor_sequential_profile_probe": True,
        "baseline_floor_calibrated_sequential_profile_probe": True,
        "baseline_floor_profile_scale_memory_probe": True,
        "baseline_floor_profile_scale_frontier_probe": True,
        "baseline_floor_profile_scale_memory_consolidation_missing_first_token_probe": True,
        "baseline_floor_profile_scale_memory_consolidation_profile_specific_missing_first_token_probe": (
            profile_specific
        ),
        "learning_rate_scale": profile_scale,
        "missing_first_token_learning_rate_scale": missing_token_scale,
        "update_shape": update_shape,
        "sequential_profile": profile,
        "sequential_profile_records": profile_batch_size,
        "sequential_profile_frontier_records": profile_frontier_records,
        "missing_first_token_records": records,
        "missing_first_token_target_profiles": target_plan.target_profiles,
        "missing_first_token_target_ids": target_plan.target_ids,
        "missing_first_token_profile_specific": profile_specific,
        "memory_consolidation_target_profiles": list(target_profiles),
    }
