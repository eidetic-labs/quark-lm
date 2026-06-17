"""Branch-diversity recovery attempts for baseline-floor stabilization."""

from __future__ import annotations

from typing import Any, Callable, Sequence

from branch_diversity_snapshots import branch_diversity_snapshot_score
from branch_diversity_snapshot_coverage import (
    branch_diversity_snapshot_preserves_target_coverage,
    branch_diversity_snapshot_target_coverage_delta,
)
import transformer_baseline_floor_branch_diversity_recovery_accounting as accounting
import transformer_baseline_floor_branch_diversity_recovery_probe as recovery_probe
import transformer_baseline_floor_branch_diversity_recovery_results as results
from replay_plan import BranchReplayRecord
from transformer_baseline_floor_branch_diversity_recovery_types import (
    BranchDiversityRecoveryResult,
)
from transformer_baseline_floor_training import (
    train_direct_answer_baseline_floor_anchor_branch_diversity,
)
from transformer_baseline_floor_profile_outcomes import (
    BaselineFloorBranchDiversityRecoveryOutcome,
    evaluate_baseline_floor_branch_diversity_recovery_outcome,
)
from transformer_baseline_floor_special_rejections import (
    record_baseline_floor_branch_diversity_recovery_rejection,
)
from transformer_direct_modes import (
    BASELINE_FLOOR_BRANCH_DIVERSITY_RECOVERY_LEARNING_RATE_SCALES,
)


def try_baseline_floor_branch_diversity_recovery(
    *,
    active: bool,
    floor_preserved: bool,
    diversity_accepted: bool,
    coverage_accepted: bool,
    profile_score: tuple[float, ...] | None,
    profile_probe_snapshot: dict[str, Any] | None,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    profile_batch: list[BranchReplayRecord],
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
    profile_frontier_records: int,
    restore_direct_update_state: Callable[[dict[str, Any], dict[str, Any]], None],
    diversity_outcome: str,
    diversity_rejection_reason: str,
    learning_rate_scales: Sequence[float] = (
        BASELINE_FLOOR_BRANCH_DIVERSITY_RECOVERY_LEARNING_RATE_SCALES
    ),
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
    evaluate_outcome: Callable[
        ..., BaselineFloorBranchDiversityRecoveryOutcome
    ] = evaluate_baseline_floor_branch_diversity_recovery_outcome,
    record_rejection: Callable[[dict[str, Any], str], None] = (
        record_baseline_floor_branch_diversity_recovery_rejection
    ),
) -> BranchDiversityRecoveryResult:
    result = results.initial_branch_diversity_recovery_result(
        floor_preserved=floor_preserved,
        profile_probe_snapshot=profile_probe_snapshot,
        profile_score=profile_score,
        diversity_outcome=diversity_outcome,
        diversity_rejection_reason=diversity_rejection_reason,
    )
    if not (
        active
        and floor_preserved
        and diversity_accepted
        and coverage_accepted
        and profile_score is not None
        and profile_probe_snapshot is not None
    ):
        return result

    accounting.record_branch_diversity_recovery_candidate(update_guard)
    base_score = profile_score
    candidate_model_payload = model.to_dict(tokenizer)
    candidate_optimizer_payload = optimizer.to_dict()
    records = len(profile_batch)
    loss_total = 0.0
    loss_count = 0
    attempted = False
    outcome = "not_attempted"
    rejection_reason = ""
    recovery_score: tuple[float, ...] | None = None
    recovery_delta: dict[str, Any] | None = None
    for recovery_learning_rate_scale in learning_rate_scales:
        restore_direct_update_state(
            candidate_model_payload,
            candidate_optimizer_payload,
        )
        attempted = True
        accounting.record_branch_diversity_recovery_attempt(update_guard, records)
        loss = train_branch_diversity(
            model,
            profile_batch,
            base_learning_rate * profile_scale * recovery_learning_rate_scale,
            negative_weight,
            positive_weight,
            contrast_weight,
            params=params,
        )
        loss_total += loss
        loss_count += 1
        probe_snapshot = recovery_probe.record_branch_diversity_recovery_probe(
            snapshot_recorder=snapshot_recorder,
            direct_step=direct_step,
            profile_scale=profile_scale,
            recovery_learning_rate_scale=recovery_learning_rate_scale,
            update_shape=update_shape,
            profile=profile,
            profile_records=records,
            profile_frontier_records=profile_frontier_records,
            recovery_records=records,
        )
        recovery_floor_preserved = preserves_target_coverage(
            probe_snapshot,
            direct_baseline,
        )
        recovery_score = snapshot_score(probe_snapshot)
        recovery_delta = target_coverage_delta(
            probe_snapshot,
            profile_probe_snapshot,
        )
        recovery_outcome = evaluate_outcome(
            floor_preserved=recovery_floor_preserved,
            recovery_score=recovery_score,
            base_score=base_score,
            coverage_delta=recovery_delta,
        )
        outcome = recovery_outcome.outcome
        rejection_reason = recovery_outcome.rejection_reason
        if recovery_outcome.accepted:
            accounting.record_branch_diversity_recovery_acceptance(update_guard)
            return results.accepted_branch_diversity_recovery_result(
                loss_total=loss_total,
                loss_count=loss_count,
                floor_preserved=recovery_floor_preserved,
                profile_probe_snapshot=probe_snapshot,
                profile_score=recovery_score,
                attempted=attempted,
                outcome=outcome,
                rejection_reason=rejection_reason,
                recovery_learning_rate_scale=recovery_learning_rate_scale,
                records=records,
                base_score=base_score,
                recovery_delta=recovery_delta,
            )
        record_rejection(update_guard, rejection_reason)

    restore_direct_update_state(
        candidate_model_payload,
        candidate_optimizer_payload,
    )
    accounting.record_branch_diversity_recovery_fallback(update_guard)
    return results.fallback_branch_diversity_recovery_result(
        loss_total=loss_total,
        loss_count=loss_count,
        floor_preserved=floor_preserved,
        profile_probe_snapshot=profile_probe_snapshot,
        profile_score=profile_score,
        diversity_outcome=diversity_outcome,
        diversity_rejection_reason=diversity_rejection_reason,
        attempted=attempted,
        outcome=outcome,
        rejection_reason=rejection_reason,
        records=records,
        base_score=base_score,
        recovery_score=recovery_score,
        recovery_delta=recovery_delta,
    )
