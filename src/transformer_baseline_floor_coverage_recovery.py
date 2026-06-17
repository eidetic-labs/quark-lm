"""Coverage recovery attempts for baseline-floor profile-scale stabilization."""

from __future__ import annotations

from typing import Any, Callable, Sequence

from branch_diversity_snapshots import branch_diversity_snapshot_score
from branch_diversity_snapshot_coverage import (
    branch_diversity_snapshot_preserves_target_coverage,
    branch_diversity_snapshot_target_coverage_delta,
)
from replay_plan import BranchReplayRecord
from transformer_baseline_floor_coverage_recovery_accounting import (
    record_branch_stable_coverage_check,
    record_coverage_recovery_acceptance,
    record_coverage_recovery_attempt,
    record_coverage_recovery_candidate,
)
from transformer_baseline_floor_coverage_recovery_attempts import (
    CoverageRecoveryAttemptState,
)
from transformer_baseline_floor_coverage_recovery_batches import (
    select_coverage_recovery_batch,
)
from transformer_baseline_floor_coverage_recovery_probes import (
    record_coverage_recovery_probe,
)
from transformer_baseline_floor_training import (
    train_direct_answer_baseline_floor_anchor_batch,
)
from transformer_baseline_floor_profile_outcomes import (
    BaselineFloorCoverageRecoveryOutcome,
    evaluate_baseline_floor_coverage_recovery_outcome,
)
from transformer_baseline_floor_coverage_recovery_results import (
    CoverageRecoveryResult,
    accepted_coverage_recovery_result,
    initial_coverage_recovery_result,
    rejected_coverage_recovery_result,
)
from transformer_baseline_floor_special_rejections import (
    record_baseline_floor_coverage_recovery_rejection,
)
from transformer_direct_modes import (
    BASELINE_FLOOR_COVERAGE_RECOVERY_LEARNING_RATE_SCALES,
)


def try_baseline_floor_coverage_recovery(
    *,
    active: bool,
    coverage_prep_accepted: bool,
    profile_base_snapshot: dict[str, Any] | None,
    profile_base_score: tuple[float, ...] | None,
    profile_score: tuple[float, ...] | None,
    profile_probe_snapshot: dict[str, Any] | None,
    coverage_delta: dict[str, Any] | None,
    coverage_outcome: str,
    coverage_rejection_reason: str,
    floor_preserved: bool,
    diversity_outcome: str,
    diversity_rejection_reason: str,
    branch_stable_active: bool,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    profile_batch: list[BranchReplayRecord],
    frontier_targets_by_profile: dict[str, set[int]],
    base_learning_rate: float,
    profile_scale: float,
    params: Any,
    direct_step: int,
    direct_baseline: dict[str, Any],
    snapshot_recorder: Any,
    update_guard: dict[str, Any],
    update_shape: str,
    profile: str,
    profile_frontier_records: int,
    restore_direct_update_state: Callable[[dict[str, Any], dict[str, Any]], None],
    learning_rate_scales: Sequence[float] = (
        BASELINE_FLOOR_COVERAGE_RECOVERY_LEARNING_RATE_SCALES
    ),
    train_anchor_batch: Callable[..., float] = (
        train_direct_answer_baseline_floor_anchor_batch
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
        ..., BaselineFloorCoverageRecoveryOutcome
    ] = evaluate_baseline_floor_coverage_recovery_outcome,
    record_rejection: Callable[..., None] = (
        record_baseline_floor_coverage_recovery_rejection
    ),
) -> CoverageRecoveryResult:
    prepared_score = profile_score
    result = initial_coverage_recovery_result(
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
    if not (active and coverage_prep_accepted and profile_base_snapshot is not None):
        return result

    record_coverage_recovery_candidate(update_guard)
    prep_model_payload = model.to_dict(tokenizer)
    prep_optimizer_payload = optimizer.to_dict()
    recovery_batch = select_coverage_recovery_batch(
        profile_batch,
        frontier_targets_by_profile,
        profile,
    )
    records = len(recovery_batch)
    attempt = CoverageRecoveryAttemptState()
    for recovery_learning_rate_scale in learning_rate_scales:
        restore_direct_update_state(prep_model_payload, prep_optimizer_payload)
        record_coverage_recovery_attempt(update_guard, records)
        loss = train_anchor_batch(
            model,
            recovery_batch,
            base_learning_rate * profile_scale * recovery_learning_rate_scale,
            params=params,
        )
        attempt.record_loss(loss)
        probe_snapshot = record_coverage_recovery_probe(
            snapshot_recorder=snapshot_recorder,
            direct_step=direct_step,
            profile_scale=profile_scale,
            recovery_learning_rate_scale=recovery_learning_rate_scale,
            update_shape=update_shape,
            profile=profile,
            profile_batch_records=len(profile_batch),
            profile_frontier_records=profile_frontier_records,
            recovery_records=records,
        )
        recovery_floor_preserved = preserves_target_coverage(
            probe_snapshot,
            direct_baseline,
        )
        recovery_score = snapshot_score(probe_snapshot)
        if branch_stable_active:
            attempt.record_branch_stable_check()
            record_branch_stable_coverage_check(update_guard)
        recovery_delta = target_coverage_delta(
            probe_snapshot,
            profile_base_snapshot,
        )
        outcome = evaluate_outcome(
            recovery_floor_preserved=recovery_floor_preserved,
            recovery_score=recovery_score,
            profile_base_score=profile_base_score,
            recovery_delta=recovery_delta,
            branch_stable_active=branch_stable_active,
            prepared_score=prepared_score,
        )
        attempt.apply_outcome(
            outcome,
            recovery_delta=recovery_delta,
            recovery_score=recovery_score,
        )
        if outcome.accepted:
            record_coverage_recovery_acceptance(
                update_guard,
                branch_stable_active=branch_stable_active,
            )
            return accepted_coverage_recovery_result(
                loss_total=attempt.loss_total,
                loss_count=attempt.loss_count,
                recovery_floor_preserved=recovery_floor_preserved,
                probe_snapshot=probe_snapshot,
                recovery_score=recovery_score,
                diversity_outcome=diversity_outcome,
                profile_base_score=profile_base_score,
                recovery_delta=recovery_delta,
                recovery_outcome=attempt.recovery_outcome,
                rejection_reason=attempt.rejection_reason,
                recovery_learning_rate_scale=recovery_learning_rate_scale,
                records=records,
                prepared_score=prepared_score,
                branch_stable_checked=attempt.branch_stable_checked,
                branch_stable_accepted=attempt.branch_stable_accepted,
                branch_stability_preserved=attempt.branch_stability_preserved,
            )
        attempt.mark_coverage_tie()
        record_rejection(
            update_guard,
            attempt.rejection_reason,
            branch_stable_active=branch_stable_active,
        )

    restore_direct_update_state(prep_model_payload, prep_optimizer_payload)
    return rejected_coverage_recovery_result(
        loss_total=attempt.loss_total,
        loss_count=attempt.loss_count,
        floor_preserved=floor_preserved,
        profile_probe_snapshot=profile_probe_snapshot,
        profile_score=profile_score,
        diversity_outcome=diversity_outcome,
        diversity_rejection_reason=diversity_rejection_reason,
        coverage_delta=coverage_delta,
        coverage_outcome=coverage_outcome,
        coverage_rejection_reason=coverage_rejection_reason,
        coverage_prep_accepted=coverage_prep_accepted,
        attempted=attempt.attempted,
        recovery_outcome=attempt.recovery_outcome,
        rejection_reason=attempt.rejection_reason,
        records=records,
        recovery_delta=attempt.recovery_delta,
        prepared_score=prepared_score,
        recovery_score=attempt.recovery_score,
        branch_stable_checked=attempt.branch_stable_checked,
        branch_stable_accepted=attempt.branch_stable_accepted,
        branch_stability_preserved=attempt.branch_stability_preserved,
    )
