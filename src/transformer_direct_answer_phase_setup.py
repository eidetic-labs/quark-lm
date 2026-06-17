"""Initialization for direct-answer phase runtime state."""

from __future__ import annotations

from typing import Any

from transformer_direct_answer_snapshot_lifecycle import (
    DirectAnswerBestSnapshotTracker,
    DirectAnswerSnapshotRecorder,
)
from transformer_direct_answer_guard_state import build_direct_answer_update_guard
from transformer_direct_answer_phase_types import DirectAnswerPhaseRuntime
from transformer_direct_answer_update_guard import direct_answer_update_parameters
from transformer_training import ShuffledTrainingCursor


def initialize_direct_answer_phase(
    *,
    args: Any,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    eval_records: dict[str, list[dict[str, Any]]],
    generation_config: Any,
    direct_setup: Any,
) -> DirectAnswerPhaseRuntime:
    snapshot_recorder = DirectAnswerSnapshotRecorder(
        model=lambda: model,
        tokenizer=lambda: tokenizer,
        eval_records=eval_records,
        branch_position=args.direct_answer_branch_position,
        max_new_chars=args.direct_answer_max_new_chars,
        snapshot_mode=args.direct_answer_snapshot_mode,
        terminator=direct_setup.direct_answer_terminator,
        generation_config=generation_config,
        history_writer=direct_setup.direct_history_writer,
    )
    baseline = snapshot_recorder.append(0, None)
    best_snapshot = DirectAnswerBestSnapshotTracker.from_baseline(
        baseline,
        model,
        tokenizer,
        optimizer,
    )
    branch_context_gate = baseline["branch_context_gate"]
    training_skipped = (
        args.direct_answer_require_branch_context_gate
        and not branch_context_gate["passed"]
    )
    skip_reason = "branch_context_gate_failed" if training_skipped else None
    steps_to_run = 0 if training_skipped else args.direct_answer_steps
    if training_skipped:
        print("skipped direct-answer training: branch context gate failed")
    params = direct_answer_update_parameters(
        model,
        args.direct_answer_train_top_layer_only,
        args.direct_answer_freeze_output_bias,
    )
    update_guard = build_direct_answer_update_guard(
        direct_answer_mode=args.direct_answer_mode,
        memory_consolidation_max_profiles=args.memory_consolidation_max_profiles,
        direct_baseline_floor_learning_rate_scales=(
            direct_setup.direct_baseline_floor_learning_rate_scales
        ),
        direct_baseline_floor_outer_learning_rate_scales=(
            direct_setup.direct_baseline_floor_outer_learning_rate_scales
        ),
        direct_baseline_floor_repair_anchors=(
            direct_setup.direct_baseline_floor_repair_anchors
        ),
        direct_baseline_floor_frontier_anchors=(
            direct_setup.direct_baseline_floor_frontier_anchors
        ),
        direct_remaining_profile_binding_target_profiles=(
            direct_setup.direct_remaining_profile_binding_target_profiles
        ),
        direct_remaining_profile_binding_source_labels=(
            direct_setup.direct_remaining_profile_binding_source_labels
        ),
        direct_replay_plan=direct_setup.direct_replay_plan,
        direct_memory_consolidation_source_plan_path=(
            direct_setup.direct_memory_consolidation_source_plan_path
        ),
        direct_memory_consolidation_source_plan_summary=(
            direct_setup.direct_memory_consolidation_source_plan_summary
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
    )
    return DirectAnswerPhaseRuntime(
        snapshot_recorder=snapshot_recorder,
        baseline=baseline,
        best_snapshot=best_snapshot,
        branch_context_gate=branch_context_gate,
        training_skipped=training_skipped,
        skip_reason=skip_reason,
        steps_to_run=steps_to_run,
        training_cursor=ShuffledTrainingCursor(
            direct_setup.direct_training_pool,
            direct_setup.direct_rng,
        ),
        params=params,
        update_guard=update_guard,
        last_snapshot=baseline,
        last_snapshot_step=0,
    )
