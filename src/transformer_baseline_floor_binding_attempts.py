"""Retry loop for collapsed-profile binding attempts."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import transformer_baseline_floor_binding_accounting as binding_accounting
import transformer_baseline_floor_binding_results as binding_results
from replay_plan import BranchReplayRecord
from transformer_baseline_floor_binding_attempt import (
    run_collapsed_profile_binding_attempt,
)
from transformer_baseline_floor_binding_types import CollapsedProfileBindingResult
from transformer_baseline_floor_profile_outcome_types import (
    BaselineFloorCollapsedProfileBindingOutcome,
)


def run_collapsed_profile_binding_attempts(
    *,
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
    floor_preserved: bool,
    profile_probe_snapshot: dict[str, Any],
    profile_score: tuple[float, ...],
    diversity_outcome: str,
    diversity_rejection_reason: str,
    owner_paraphrase_binding_active: bool,
    owner_paraphrase_preserved_profiles: Sequence[str],
    owner_paraphrase_binding_preservation_delta: dict[str, Any] | None,
    memory_consolidation_active: bool,
    memory_consolidation_target_profiles: Sequence[str],
    learning_rate_scales: Sequence[float],
    target_profiles: list[str],
    train_branch_diversity: Callable[..., float],
    preserves_target_coverage: Callable[[dict[str, Any], dict[str, Any]], bool],
    snapshot_score: Callable[[dict[str, Any]], tuple[float, ...]],
    target_coverage_delta: Callable[
        [dict[str, Any], dict[str, Any]], dict[str, Any]
    ],
    profile_diversity_delta: Callable[
        [dict[str, Any], dict[str, Any], Sequence[str]], dict[str, Any]
    ],
    evaluate_outcome: Callable[..., BaselineFloorCollapsedProfileBindingOutcome],
    record_rejection: Callable[[dict[str, Any], str], None],
) -> CollapsedProfileBindingResult:
    binding_accounting.record_collapsed_profile_binding_candidate(update_guard)
    base_score = profile_score
    candidate_model_payload = model.to_dict(tokenizer)
    candidate_optimizer_payload = optimizer.to_dict()
    records = len(profile_batch)
    loss_total = 0.0
    loss_count = 0
    attempted = False
    outcome = "not_attempted"
    rejection_reason = ""
    binding_score: tuple[float, ...] | None = None
    binding_delta: dict[str, Any] | None = None
    current_preservation_delta = owner_paraphrase_binding_preservation_delta

    for binding_learning_rate_scale in learning_rate_scales:
        restore_direct_update_state(candidate_model_payload, candidate_optimizer_payload)
        attempted = True
        binding_accounting.record_collapsed_profile_binding_attempt(
            update_guard,
            records,
        )
        attempt_result = run_collapsed_profile_binding_attempt(
            model=model,
            profile_batch=profile_batch,
            base_learning_rate=base_learning_rate,
            profile_scale=profile_scale,
            binding_learning_rate_scale=binding_learning_rate_scale,
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
            target_profiles=target_profiles,
            profile_probe_snapshot=profile_probe_snapshot,
            base_score=base_score,
            memory_consolidation_active=memory_consolidation_active,
            memory_consolidation_target_profiles=memory_consolidation_target_profiles,
            owner_paraphrase_binding_active=owner_paraphrase_binding_active,
            owner_paraphrase_preserved_profiles=owner_paraphrase_preserved_profiles,
            current_preservation_delta=current_preservation_delta,
            train_branch_diversity=train_branch_diversity,
            preserves_target_coverage=preserves_target_coverage,
            snapshot_score=snapshot_score,
            target_coverage_delta=target_coverage_delta,
            profile_diversity_delta=profile_diversity_delta,
            evaluate_outcome=evaluate_outcome,
        )
        loss_total += attempt_result.loss
        loss_count += 1
        binding_score = attempt_result.score
        binding_delta = attempt_result.delta
        current_preservation_delta = attempt_result.preservation_delta
        outcome = attempt_result.outcome
        rejection_reason = attempt_result.rejection_reason
        if attempt_result.accepted:
            binding_accounting.record_collapsed_profile_binding_acceptance(update_guard)
            return binding_results.accepted_collapsed_profile_binding_result(
                attempt_result=attempt_result,
                loss_total=loss_total,
                loss_count=loss_count,
                attempted=attempted,
                binding_learning_rate_scale=binding_learning_rate_scale,
                records=records,
                target_profiles=target_profiles,
                base_score=base_score,
                diversity_outcome=diversity_outcome,
            )
        record_rejection(update_guard, rejection_reason)

    restore_direct_update_state(candidate_model_payload, candidate_optimizer_payload)
    binding_accounting.record_collapsed_profile_binding_fallback(update_guard)
    return binding_results.fallback_collapsed_profile_binding_result(
        loss_total=loss_total,
        loss_count=loss_count,
        floor_preserved=floor_preserved,
        profile_probe_snapshot=profile_probe_snapshot,
        profile_score=profile_score,
        diversity_outcome=diversity_outcome,
        diversity_rejection_reason=diversity_rejection_reason,
        owner_paraphrase_binding_preservation_delta=current_preservation_delta,
        attempted=attempted,
        outcome=outcome,
        rejection_reason=rejection_reason,
        records=records,
        target_profiles=target_profiles,
        base_score=base_score,
        binding_score=binding_score,
        binding_delta=binding_delta,
    )
