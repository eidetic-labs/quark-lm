"""Direct-answer stage orchestration for transformer answer training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformer_answer_direct_stage_adaptive import (
    build_adaptive_baseline_floor_trainer,
)
from transformer_answer_direct_stage_state import DirectAnswerStageState
from transformer_answer_direct_stage_updaters import (
    build_baseline_anchored_prompt_updater,
    build_stabilization_context,
    build_stabilization_trainer,
)
from transformer_direct_answer_phase import (
    complete_direct_answer_phase,
    initialize_direct_answer_phase,
    run_direct_answer_training_loop,
)
from transformer_direct_answer_setup import prepare_direct_answer_run_setup


@dataclass
class TransformerDirectAnswerStageResult:
    model: Any
    tokenizer: Any
    optimizer: Any
    training_plan: dict[str, Any]
    last_snapshot: dict[str, Any]
    post_direct_candidate_snapshot: dict[str, Any] | None
    post_direct_candidate_snapshot_skipped: bool
    direct_answer_metrics: dict[str, Any] | None


def run_transformer_direct_answer_stage(
    *,
    args: Any,
    model_class: type[Any],
    setup: Any,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    training_plan: dict[str, Any],
    last_snapshot: dict[str, Any],
    snapshot: Any,
) -> TransformerDirectAnswerStageResult:
    direct_setup = prepare_direct_answer_run_setup(
        args=args,
        model=model,
        tokenizer=tokenizer,
        examples=setup.examples,
        training_plan=training_plan,
        training_plan_path=setup.training_plan_path,
    )
    training_plan = direct_setup.training_plan
    direct_runtime = initialize_direct_answer_phase(
        args=args,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        eval_records=setup.eval_records,
        generation_config=setup.generation_config,
        direct_setup=direct_setup,
    )
    direct_params = direct_runtime.params

    stage_state = DirectAnswerStageState(
        args=args,
        model_class=model_class,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        params=direct_params,
    )
    baseline_anchored_prompt_updater = build_baseline_anchored_prompt_updater(
        args=args,
        direct_setup=direct_setup,
        direct_runtime=direct_runtime,
        stage_state=stage_state,
    )
    stabilization_context = build_stabilization_context(
        args=args,
        direct_setup=direct_setup,
        direct_runtime=direct_runtime,
        stage_state=stage_state,
    )
    train_baseline_floor_stabilization_update = build_stabilization_trainer(
        stabilization_context
    )
    train_adaptive_baseline_floor_update = build_adaptive_baseline_floor_trainer(
        args=args,
        direct_setup=direct_setup,
        direct_runtime=direct_runtime,
        stage_state=stage_state,
        train_stabilization_update=train_baseline_floor_stabilization_update,
        train_baseline_anchored_prompt=baseline_anchored_prompt_updater.train,
    )

    def restore_direct_update_state(
        model_payload: dict[str, Any],
        optimizer_payload: dict[str, Any],
    ) -> None:
        restore_stage_state_and_rebind_recorder(
            stage_state,
            direct_runtime.snapshot_recorder,
            model_payload,
            optimizer_payload,
        )

    loop_result = run_direct_answer_training_loop(
        args=args,
        model=stage_state.model,
        tokenizer=stage_state.tokenizer,
        optimizer=stage_state.optimizer,
        direct_lessons=direct_setup.direct_lessons,
        direct_training_pool=direct_setup.direct_training_pool,
        direct_training_cursor=direct_runtime.training_cursor,
        direct_rng=direct_setup.direct_rng,
        direct_steps_to_run=direct_runtime.steps_to_run,
        direct_answer_terminator=direct_setup.direct_answer_terminator,
        direct_params=stage_state.params,
        direct_answer_baseline_floor_update_gate_active=(
            direct_setup.direct_answer_baseline_floor_update_gate_active
        ),
        direct_answer_baseline_floor_adaptive_updates_active=(
            direct_setup.direct_answer_baseline_floor_adaptive_updates_active
        ),
        direct_answer_update_guard=direct_runtime.update_guard,
        direct_baseline=direct_runtime.baseline,
        last_direct_snapshot=direct_runtime.last_snapshot,
        last_direct_snapshot_step=direct_runtime.last_snapshot_step,
        direct_snapshot_recorder=direct_runtime.snapshot_recorder,
        best_direct_snapshot=direct_runtime.best_snapshot,
        train_adaptive_baseline_floor_update=train_adaptive_baseline_floor_update,
        train_baseline_anchored_prompt=baseline_anchored_prompt_updater.train,
        restore_direct_update_state=restore_direct_update_state,
    )
    direct_phase = complete_direct_answer_phase(
        args=args,
        model_class=model_class,
        optimizer_class=type(stage_state.optimizer),
        model=stage_state.model,
        tokenizer=stage_state.tokenizer,
        optimizer=stage_state.optimizer,
        direct_snapshot_recorder=direct_runtime.snapshot_recorder,
        best_direct_snapshot=direct_runtime.best_snapshot,
        last_direct_snapshot=loop_result.last_snapshot,
        last_direct_snapshot_step=loop_result.last_snapshot_step,
        snapshot=snapshot,
        direct_setup=direct_setup,
        direct_steps_to_run=direct_runtime.steps_to_run,
        direct_training_example_count=len(direct_setup.direct_training_pool),
        direct_answer_update_guard=direct_runtime.update_guard,
        direct_baseline=direct_runtime.baseline,
        direct_answer_training_skipped=direct_runtime.training_skipped,
        direct_answer_skip_reason=direct_runtime.skip_reason,
        branch_context_gate=direct_runtime.branch_context_gate,
        generation_config=setup.generation_config,
        direct_answer_terminator=direct_setup.direct_answer_terminator,
        context_coverage=setup.context_coverage,
        routing_repair_batch_evidence=loop_result.routing_repair_batch_evidence,
    )
    return TransformerDirectAnswerStageResult(
        model=direct_phase.model,
        tokenizer=direct_phase.tokenizer,
        optimizer=direct_phase.optimizer,
        training_plan=training_plan,
        last_snapshot=direct_phase.last_snapshot,
        post_direct_candidate_snapshot=direct_phase.post_direct_candidate_snapshot,
        post_direct_candidate_snapshot_skipped=(
            direct_phase.post_direct_candidate_snapshot_skipped
        ),
        direct_answer_metrics=direct_phase.metrics,
    )


def restore_stage_state_and_rebind_recorder(
    stage_state: DirectAnswerStageState,
    snapshot_recorder: Any,
    model_payload: dict[str, Any],
    optimizer_payload: dict[str, Any],
) -> tuple[Any, Any, Any, Any]:
    stage_state.restore(model_payload, optimizer_payload)
    snapshot_recorder.model = lambda: stage_state.model
    snapshot_recorder.tokenizer = lambda: stage_state.tokenizer
    return (
        stage_state.model,
        stage_state.tokenizer,
        stage_state.optimizer,
        stage_state.params,
    )
