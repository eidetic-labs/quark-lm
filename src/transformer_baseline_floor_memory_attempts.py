"""Attempt loop for missing-first-token memory consolidation."""

from __future__ import annotations

from typing import Any, Callable, Sequence

from branch_diversity_snapshots import (
    branch_diversity_snapshot_profile_diversity_delta,
    branch_diversity_snapshot_score,
)
from branch_diversity_snapshot_coverage import (
    branch_diversity_snapshot_preserves_target_coverage,
    branch_diversity_snapshot_target_coverage_delta,
)
from transformer_baseline_floor_memory_candidate import MissingFirstTokenCandidate
from transformer_baseline_floor_memory_results import (
    MissingFirstTokenResult,
    accepted_missing_first_token_result,
    fallback_missing_first_token_result,
    missing_first_token_probe_metadata,
)
from transformer_baseline_floor_profile_outcomes import (
    BaselineFloorMissingFirstTokenOutcome,
    evaluate_baseline_floor_missing_first_token_outcome,
)
from transformer_baseline_floor_special_rejections import (
    record_baseline_floor_missing_first_token_rejection,
)
from transformer_baseline_floor_training import (
    train_direct_answer_baseline_floor_anchor_branch_diversity,
)


def run_missing_first_token_candidate_attempts(
    *,
    base_result: MissingFirstTokenResult,
    candidate: MissingFirstTokenCandidate,
    profile_score: tuple[float, ...],
    profile_probe_snapshot: dict[str, Any],
    coverage_delta: dict[str, Any] | None,
    profile_base_snapshot: dict[str, Any] | None,
    model: Any,
    base_learning_rate: float,
    profile_scale: float,
    negative_weight: float,
    positive_weight: float,
    contrast_weight: float,
    params: Any,
    direct_step: int,
    direct_baseline: dict[str, Any],
    snapshot_recorder: Any,
    update_guard: dict[str, Any],
    update_shape: str,
    profile: str,
    profile_batch_size: int,
    profile_frontier_records: int,
    target_profiles: Sequence[str],
    profile_specific: bool,
    restore_direct_update_state: Callable[[dict[str, Any], dict[str, Any]], None],
    diversity_outcome: str,
    learning_rate_scales: Sequence[float],
    train_branch_diversity: Callable[..., float] = (
        train_direct_answer_baseline_floor_anchor_branch_diversity
    ),
    preserves_target_coverage: Callable[
        [dict[str, Any], dict[str, Any]], bool
    ] = branch_diversity_snapshot_preserves_target_coverage,
    snapshot_score: Callable[[dict[str, Any]], tuple[float, ...]] = (
        branch_diversity_snapshot_score
    ),
    target_coverage_delta: Callable[
        [dict[str, Any], dict[str, Any]], dict[str, Any]
    ] = branch_diversity_snapshot_target_coverage_delta,
    profile_diversity_delta: Callable[
        [dict[str, Any], dict[str, Any], Sequence[str]], dict[str, Any]
    ] = branch_diversity_snapshot_profile_diversity_delta,
    evaluate_outcome: Callable[
        ..., BaselineFloorMissingFirstTokenOutcome
    ] = evaluate_baseline_floor_missing_first_token_outcome,
    record_rejection: Callable[[dict[str, Any], str], None] = (
        record_baseline_floor_missing_first_token_rejection
    ),
) -> MissingFirstTokenResult:
    update_guard[
        "profile_scale_memory_consolidation_missing_first_token_candidates"
    ] += 1
    base_score = profile_score
    loss_total = 0.0
    loss_count = 0
    attempted = False
    outcome = "not_attempted"
    rejection_reason = ""
    token_score: tuple[float, ...] | None = None
    token_delta: dict[str, Any] | None = None
    for missing_token_scale in learning_rate_scales:
        restore_direct_update_state(
            candidate.model_payload,
            candidate.optimizer_payload,
        )
        attempted = True
        update_guard["profile_scale_memory_consolidation_missing_first_token_attempts"] += 1
        update_guard["profile_scale_memory_consolidation_missing_first_token_records"] += (
            candidate.records
        )
        loss = train_branch_diversity(
            model,
            candidate.missing_batch,
            base_learning_rate * profile_scale * missing_token_scale,
            negative_weight,
            positive_weight,
            contrast_weight,
            params=params,
        )
        loss_total += loss
        loss_count += 1
        probe_snapshot = snapshot_recorder.record(
            direct_step,
            None,
            missing_first_token_probe_metadata(
                profile_scale=profile_scale,
                missing_token_scale=missing_token_scale,
                update_shape=update_shape,
                profile=profile,
                profile_batch_size=profile_batch_size,
                profile_frontier_records=profile_frontier_records,
                records=candidate.records,
                target_plan=candidate.target_plan,
                profile_specific=profile_specific,
                target_profiles=target_profiles,
            ),
        )
        token_floor_preserved = preserves_target_coverage(
            probe_snapshot,
            direct_baseline,
        )
        token_score = snapshot_score(probe_snapshot)
        token_coverage_delta = target_coverage_delta(
            probe_snapshot,
            profile_probe_snapshot,
        )
        token_profile_delta = profile_diversity_delta(
            probe_snapshot,
            profile_probe_snapshot,
            candidate.target_plan.target_profiles,
        )
        token_delta = {
            "coverage_delta": token_coverage_delta,
            "profile_delta": token_profile_delta,
        }
        token_outcome = evaluate_outcome(
            floor_preserved=token_floor_preserved,
            token_score=token_score,
            base_score=base_score,
            coverage_delta=token_coverage_delta,
            profile_delta=token_profile_delta,
        )
        outcome = token_outcome.outcome
        rejection_reason = token_outcome.rejection_reason
        if token_outcome.accepted:
            accepted_coverage_delta = coverage_delta
            if profile_base_snapshot is not None:
                accepted_coverage_delta = target_coverage_delta(
                    probe_snapshot,
                    profile_base_snapshot,
                )
            accepted_diversity_outcome = diversity_outcome
            if token_score > base_score:
                accepted_diversity_outcome = "improved"
            update_guard[
                "profile_scale_memory_consolidation_missing_first_token_acceptances"
            ] += 1
            return accepted_missing_first_token_result(
                loss_total=loss_total,
                loss_count=loss_count,
                floor_preserved=token_floor_preserved,
                profile_probe_snapshot=probe_snapshot,
                profile_score=token_score,
                diversity_outcome=accepted_diversity_outcome,
                coverage_delta=accepted_coverage_delta,
                attempted=attempted,
                outcome=outcome,
                rejection_reason=rejection_reason,
                learning_rate_scale=missing_token_scale,
                records=candidate.records,
                target_plan=candidate.target_plan,
                base_score=base_score,
                delta=token_delta,
            )
        record_rejection(update_guard, rejection_reason)

    restore_direct_update_state(candidate.model_payload, candidate.optimizer_payload)
    update_guard[
        "profile_scale_memory_consolidation_missing_first_token_fallback_acceptances"
    ] += 1
    return fallback_missing_first_token_result(
        base_result=base_result,
        loss_total=loss_total,
        loss_count=loss_count,
        attempted=attempted,
        outcome=outcome,
        rejection_reason=rejection_reason,
        records=candidate.records,
        target_plan=candidate.target_plan,
        base_score=base_score,
        score=token_score,
        delta=token_delta,
    )
