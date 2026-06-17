"""Completion and metrics assembly for the direct-answer phase."""

from __future__ import annotations

from typing import Any, Callable

from transformer_answer_snapshots import finalize_direct_answer_snapshots
from transformer_direct_answer_metrics import build_direct_answer_metrics
from transformer_direct_answer_phase_types import DirectAnswerPhaseResult


def complete_direct_answer_phase(
    *,
    args: Any,
    model_class: type[Any],
    optimizer_class: type[Any],
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    direct_snapshot_recorder: Any,
    best_direct_snapshot: Any,
    last_direct_snapshot: dict[str, Any],
    last_direct_snapshot_step: int,
    snapshot: Callable[[int, float | None], dict[str, Any]],
    direct_setup: Any,
    direct_steps_to_run: int,
    direct_training_example_count: int,
    direct_answer_update_guard: dict[str, Any],
    direct_baseline: dict[str, Any],
    direct_answer_training_skipped: bool,
    direct_answer_skip_reason: str | None,
    branch_context_gate: dict[str, Any],
    generation_config: Any,
    direct_answer_terminator: str,
    context_coverage: dict[str, Any],
    finalize_snapshots: Callable[..., Any] = finalize_direct_answer_snapshots,
    build_metrics: Callable[..., dict[str, Any]] = build_direct_answer_metrics,
) -> DirectAnswerPhaseResult:
    snapshot_finalization = finalize_snapshots(
        direct_answer_steps=args.direct_answer_steps,
        restore_best_branch_snapshot=args.direct_answer_restore_best_branch_snapshot,
        model_class=model_class,
        optimizer_class=optimizer_class,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        recorder=direct_snapshot_recorder,
        best_snapshot=best_direct_snapshot,
        last_snapshot=last_direct_snapshot,
        last_snapshot_step=last_direct_snapshot_step,
    )
    model = snapshot_finalization.model
    tokenizer = snapshot_finalization.tokenizer
    optimizer = snapshot_finalization.optimizer
    last_direct_snapshot = snapshot_finalization.last_snapshot

    post_direct_candidate_snapshot_skipped = args.skip_post_direct_snapshot
    post_direct_candidate_snapshot: dict[str, Any] | None = None
    if post_direct_candidate_snapshot_skipped:
        print("skipped post-direct candidate snapshot")
    else:
        post_direct_candidate_snapshot = snapshot(
            args.steps + args.direct_answer_steps,
            None,
        )

    metrics = build_metrics(
        args=args,
        direct_history_path=direct_setup.direct_history_path,
        direct_steps_to_run=direct_steps_to_run,
        direct_training_example_count=direct_training_example_count,
        direct_profile_aware_targets=direct_setup.direct_profile_aware_targets,
        direct_replay_plan_path=direct_setup.direct_replay_plan_path,
        direct_replay_plan=direct_setup.direct_replay_plan,
        direct_replay_prediction_overrides=(
            direct_setup.direct_replay_prediction_overrides
        ),
        direct_replay_prediction_anchors_active=(
            direct_setup.direct_replay_prediction_anchors_active
        ),
        direct_memory_consolidation_source_plan_path=(
            direct_setup.direct_memory_consolidation_source_plan_path
        ),
        direct_memory_consolidation_target_profiles=(
            direct_setup.direct_memory_consolidation_target_profiles
        ),
        direct_memory_consolidation_top_priority_profiles=(
            direct_setup.direct_memory_consolidation_top_priority_profiles
        ),
        direct_memory_consolidation_collapsed_memory_backed_profiles=(
            direct_setup.direct_memory_consolidation_collapsed_memory_backed_profiles
        ),
        direct_memory_consolidation_missing_first_token_values=(
            direct_setup.direct_memory_consolidation_missing_first_token_values
        ),
        direct_memory_consolidation_missing_first_token_ids=(
            direct_setup.direct_memory_consolidation_missing_first_token_ids
        ),
        direct_memory_consolidation_profile_specific_missing_first_token_target_map=(
            direct_setup.direct_memory_consolidation_profile_specific_missing_first_token_target_map
        ),
        direct_answer_update_guard=direct_answer_update_guard,
        direct_answer_restored_best_branch_snapshot=(
            snapshot_finalization.restored_best_branch_snapshot
        ),
        best_direct_snapshot_step=best_direct_snapshot.step,
        best_direct_snapshot_score=best_direct_snapshot.score,
        direct_baseline=direct_baseline,
        direct_answer_training_skipped=direct_answer_training_skipped,
        direct_answer_skip_reason=direct_answer_skip_reason,
        branch_context_gate=branch_context_gate,
        post_direct_candidate_snapshot_skipped=post_direct_candidate_snapshot_skipped,
        generation_config=generation_config,
        direct_answer_terminator=direct_answer_terminator,
        context_coverage=context_coverage,
        last_direct_snapshot=last_direct_snapshot,
    )
    return DirectAnswerPhaseResult(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        last_snapshot=last_direct_snapshot,
        post_direct_candidate_snapshot=post_direct_candidate_snapshot,
        post_direct_candidate_snapshot_skipped=post_direct_candidate_snapshot_skipped,
        metrics=metrics,
    )
