"""Baseline-floor memory-consolidation attempt helpers."""

from __future__ import annotations

import random
from typing import Any, Callable, Sequence

from branch_diversity_snapshots import (
    branch_diversity_snapshot_preserves_target_coverage,
    branch_diversity_snapshot_profile_diversity_delta,
    branch_diversity_snapshot_score,
    branch_diversity_snapshot_target_coverage_delta,
)
from replay_plan import BranchReplayRecord
from transformer_baseline_floor_training import (
    train_direct_answer_baseline_floor_anchor_branch_diversity,
)
from transformer_baseline_floor_profile_outcomes import (
    BaselineFloorMissingFirstTokenOutcome,
    evaluate_baseline_floor_missing_first_token_outcome,
)
from transformer_baseline_floor_special_rejections import (
    record_baseline_floor_missing_first_token_rejection,
)
from transformer_baseline_floor_memory_candidate import (
    prepare_missing_first_token_candidate,
)
from transformer_baseline_floor_memory_attempts import (
    run_missing_first_token_candidate_attempts,
)
from transformer_baseline_floor_memory_results import (
    MissingFirstTokenResult,
    initial_missing_first_token_result,
    missing_first_token_target_result,
)
from transformer_direct_modes import (
    BASELINE_FLOOR_MISSING_FIRST_TOKEN_LEARNING_RATE_SCALES,
)
from transformer_memory_plan_helpers import (
    MissingFirstTokenTargetPlan,
    missing_first_token_anchor_batch,
    plan_missing_first_token_targets,
)


def try_baseline_floor_missing_first_token_consolidation(
    *,
    active: bool,
    memory_consolidation_prioritized: bool,
    floor_preserved: bool,
    diversity_accepted: bool,
    coverage_accepted: bool,
    profile_score: tuple[float, ...] | None,
    profile_probe_snapshot: dict[str, Any] | None,
    coverage_delta: dict[str, Any] | None,
    coverage_outcome: str,
    coverage_rejection_reason: str,
    profile_base_snapshot: dict[str, Any] | None,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    profile_batch: list[BranchReplayRecord],
    rng: random.Random,
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
    target_profiles: Sequence[str],
    missing_first_token_ids_by_profile: dict[str, list[int]],
    profile_specific: bool,
    restore_direct_update_state: Callable[[dict[str, Any], dict[str, Any]], None],
    diversity_outcome: str,
    diversity_rejection_reason: str,
    learning_rate_scales: Sequence[float] = (
        BASELINE_FLOOR_MISSING_FIRST_TOKEN_LEARNING_RATE_SCALES
    ),
    plan_targets: Callable[..., MissingFirstTokenTargetPlan] = (
        plan_missing_first_token_targets
    ),
    select_anchor_batch: Callable[
        [list[BranchReplayRecord], set[int], random.Random, int],
        list[BranchReplayRecord],
    ] = missing_first_token_anchor_batch,
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
    result = initial_missing_first_token_result(
        floor_preserved=floor_preserved,
        profile_probe_snapshot=profile_probe_snapshot,
        profile_score=profile_score,
        diversity_outcome=diversity_outcome,
        diversity_rejection_reason=diversity_rejection_reason,
        coverage_delta=coverage_delta,
        coverage_outcome=coverage_outcome,
        coverage_rejection_reason=coverage_rejection_reason,
    )
    if not (
        active
        and memory_consolidation_prioritized
        and floor_preserved
        and diversity_accepted
        and coverage_accepted
        and profile_score is not None
        and profile_probe_snapshot is not None
        and missing_first_token_ids_by_profile
    ):
        return result

    candidate = prepare_missing_first_token_candidate(
        profile,
        target_profiles,
        missing_first_token_ids_by_profile,
        profile_specific,
        profile_batch,
        rng,
        model,
        tokenizer,
        optimizer,
        plan_targets,
        select_anchor_batch,
    )
    if not candidate.missing_batch:
        return missing_first_token_target_result(result, candidate.target_plan)

    return run_missing_first_token_candidate_attempts(
        base_result=result,
        candidate=candidate,
        profile_score=profile_score,
        profile_probe_snapshot=profile_probe_snapshot,
        coverage_delta=coverage_delta,
        profile_base_snapshot=profile_base_snapshot,
        model=model,
        base_learning_rate=base_learning_rate,
        profile_scale=profile_scale,
        negative_weight=negative_weight,
        positive_weight=positive_weight,
        contrast_weight=contrast_weight,
        params=params,
        direct_step=direct_step,
        direct_baseline=direct_baseline,
        snapshot_recorder=snapshot_recorder,
        update_guard=update_guard,
        update_shape=update_shape,
        profile=profile,
        profile_batch_size=len(profile_batch),
        profile_frontier_records=profile_frontier_records,
        target_profiles=target_profiles,
        profile_specific=profile_specific,
        restore_direct_update_state=restore_direct_update_state,
        diversity_outcome=diversity_outcome,
        learning_rate_scales=learning_rate_scales,
        train_branch_diversity=train_branch_diversity,
        preserves_target_coverage=preserves_target_coverage,
        snapshot_score=snapshot_score,
        target_coverage_delta=target_coverage_delta,
        profile_diversity_delta=profile_diversity_delta,
        evaluate_outcome=evaluate_outcome,
        record_rejection=record_rejection,
    )
