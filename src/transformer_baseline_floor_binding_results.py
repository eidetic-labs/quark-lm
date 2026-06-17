"""Collapsed profile binding result builders."""

from __future__ import annotations

from typing import Any

from transformer_baseline_floor_binding_attempt import (
    CollapsedProfileBindingAttemptResult,
)
from transformer_baseline_floor_binding_types import CollapsedProfileBindingResult


def initial_collapsed_profile_binding_result(
    *,
    floor_preserved: bool,
    profile_probe_snapshot: dict[str, Any] | None,
    profile_score: tuple[float, ...] | None,
    diversity_outcome: str,
    diversity_rejection_reason: str,
    owner_paraphrase_binding_preservation_delta: dict[str, Any] | None,
) -> CollapsedProfileBindingResult:
    return CollapsedProfileBindingResult(
        loss_total=0.0,
        loss_count=0,
        floor_preserved=floor_preserved,
        profile_probe_snapshot=profile_probe_snapshot,
        profile_score=profile_score,
        diversity_outcome=diversity_outcome,
        diversity_rejection_reason=diversity_rejection_reason,
        owner_paraphrase_binding_preservation_delta=(
            owner_paraphrase_binding_preservation_delta
        ),
        target_profiles=[],
    )


def accepted_collapsed_profile_binding_result(
    *,
    attempt_result: CollapsedProfileBindingAttemptResult,
    loss_total: float,
    loss_count: int,
    attempted: bool,
    binding_learning_rate_scale: float,
    records: int,
    target_profiles: list[str],
    base_score: tuple[float, ...],
    diversity_outcome: str,
) -> CollapsedProfileBindingResult:
    accepted_diversity_outcome = diversity_outcome
    if attempt_result.score > base_score:
        accepted_diversity_outcome = "improved"
    return CollapsedProfileBindingResult(
        loss_total=loss_total,
        loss_count=loss_count,
        floor_preserved=attempt_result.floor_preserved,
        profile_probe_snapshot=attempt_result.probe_snapshot,
        profile_score=attempt_result.score,
        diversity_outcome=accepted_diversity_outcome,
        diversity_rejection_reason="",
        owner_paraphrase_binding_preservation_delta=attempt_result.preservation_delta,
        attempted=attempted,
        accepted=True,
        outcome=attempt_result.outcome,
        rejection_reason=attempt_result.rejection_reason,
        learning_rate_scale=binding_learning_rate_scale,
        records=records,
        target_profiles=target_profiles,
        base_score=base_score,
        score=attempt_result.score,
        delta=attempt_result.delta,
    )


def fallback_collapsed_profile_binding_result(
    *,
    loss_total: float,
    loss_count: int,
    floor_preserved: bool,
    profile_probe_snapshot: dict[str, Any] | None,
    profile_score: tuple[float, ...] | None,
    diversity_outcome: str,
    diversity_rejection_reason: str,
    owner_paraphrase_binding_preservation_delta: dict[str, Any] | None,
    attempted: bool,
    outcome: str,
    rejection_reason: str,
    records: int,
    target_profiles: list[str],
    base_score: tuple[float, ...],
    binding_score: tuple[float, ...] | None,
    binding_delta: dict[str, Any] | None,
) -> CollapsedProfileBindingResult:
    return CollapsedProfileBindingResult(
        loss_total=loss_total,
        loss_count=loss_count,
        floor_preserved=floor_preserved,
        profile_probe_snapshot=profile_probe_snapshot,
        profile_score=profile_score,
        diversity_outcome=diversity_outcome,
        diversity_rejection_reason=diversity_rejection_reason,
        owner_paraphrase_binding_preservation_delta=(
            owner_paraphrase_binding_preservation_delta
        ),
        attempted=attempted,
        accepted=False,
        outcome=outcome,
        rejection_reason=rejection_reason,
        records=records,
        target_profiles=target_profiles,
        base_score=base_score,
        score=binding_score,
        delta=binding_delta,
    )
