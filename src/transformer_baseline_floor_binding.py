"""Baseline-floor binding attempt helpers."""

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
import transformer_baseline_floor_binding_results as binding_results
from replay_plan import BranchReplayRecord
from transformer_baseline_floor_binding_targets import (
    select_collapsed_profile_binding_targets,
)
from transformer_baseline_floor_binding_attempts import (
    run_collapsed_profile_binding_attempts,
)
from transformer_baseline_floor_binding_types import CollapsedProfileBindingResult
from transformer_baseline_floor_training import (
    train_direct_answer_baseline_floor_anchor_branch_diversity,
)
from transformer_baseline_floor_recovery_outcomes import (
    BaselineFloorCollapsedProfileBindingOutcome,
    evaluate_baseline_floor_collapsed_profile_binding_outcome,
)
from transformer_baseline_floor_special_rejections import (
    record_baseline_floor_collapsed_profile_binding_rejection,
)
from transformer_direct_modes import (
    BASELINE_FLOOR_COLLAPSED_PROFILE_BINDING_LEARNING_RATE_SCALES,
)


def try_baseline_floor_collapsed_profile_binding(
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
    owner_paraphrase_binding_active: bool,
    owner_paraphrase_target_profiles: Sequence[str],
    owner_paraphrase_preserved_profiles: Sequence[str],
    owner_paraphrase_binding_preservation_delta: dict[str, Any] | None,
    memory_consolidation_active: bool,
    memory_consolidation_target_profiles: Sequence[str],
    learning_rate_scales: Sequence[float] = (
        BASELINE_FLOOR_COLLAPSED_PROFILE_BINDING_LEARNING_RATE_SCALES
    ),
    select_binding_targets: Callable[..., list[str]] = (
        select_collapsed_profile_binding_targets
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
    profile_diversity_delta: Callable[
        [dict[str, Any], dict[str, Any], Sequence[str]], dict[str, Any]
    ] = branch_diversity_snapshot_profile_diversity_delta,
    evaluate_outcome: Callable[
        ..., BaselineFloorCollapsedProfileBindingOutcome
    ] = evaluate_baseline_floor_collapsed_profile_binding_outcome,
    record_rejection: Callable[[dict[str, Any], str], None] = (
        record_baseline_floor_collapsed_profile_binding_rejection
    ),
) -> CollapsedProfileBindingResult:
    result = binding_results.initial_collapsed_profile_binding_result(
        floor_preserved=floor_preserved,
        profile_probe_snapshot=profile_probe_snapshot,
        profile_score=profile_score,
        diversity_outcome=diversity_outcome,
        diversity_rejection_reason=diversity_rejection_reason,
        owner_paraphrase_binding_preservation_delta=(
            owner_paraphrase_binding_preservation_delta
        ),
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

    target_profiles = select_binding_targets(
        profile_probe_snapshot,
        memory_consolidation_active=memory_consolidation_active,
        memory_consolidation_target_profiles=memory_consolidation_target_profiles,
        owner_paraphrase_binding_active=owner_paraphrase_binding_active,
        owner_paraphrase_target_profiles=owner_paraphrase_target_profiles,
    )
    if not target_profiles:
        return result

    return run_collapsed_profile_binding_attempts(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        profile_batch=profile_batch,
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
        profile_frontier_records=profile_frontier_records,
        restore_direct_update_state=restore_direct_update_state,
        floor_preserved=floor_preserved,
        profile_probe_snapshot=profile_probe_snapshot,
        profile_score=profile_score,
        diversity_outcome=diversity_outcome,
        diversity_rejection_reason=diversity_rejection_reason,
        owner_paraphrase_binding_active=owner_paraphrase_binding_active,
        owner_paraphrase_preserved_profiles=owner_paraphrase_preserved_profiles,
        owner_paraphrase_binding_preservation_delta=(
            owner_paraphrase_binding_preservation_delta
        ),
        memory_consolidation_active=memory_consolidation_active,
        memory_consolidation_target_profiles=memory_consolidation_target_profiles,
        learning_rate_scales=learning_rate_scales,
        target_profiles=target_profiles,
        train_branch_diversity=train_branch_diversity,
        preserves_target_coverage=preserves_target_coverage,
        snapshot_score=snapshot_score,
        target_coverage_delta=target_coverage_delta,
        profile_diversity_delta=profile_diversity_delta,
        evaluate_outcome=evaluate_outcome,
        record_rejection=record_rejection,
    )
