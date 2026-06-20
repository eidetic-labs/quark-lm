"""Training-loop execution for the direct-answer phase."""

from __future__ import annotations

from typing import Any, Callable

from transformer_direct_answer_mode_dispatch import train_direct_answer_mode_step
from transformer_direct_answer_frontier_update_guard import (
    apply_direct_frontier_update_guard_probe,
    direct_answer_frontier_update_guard_active,
)
from transformer_direct_answer_phase_types import DirectAnswerLoopResult
from transformer_direct_answer_update_guard import apply_direct_update_guard_probe
from transformer_routing_repair_batch_evidence import (
    record_routing_repair_batch_step,
    routing_repair_batch_evidence_enabled,
    routing_repair_batch_evidence_summary,
)
from transformer_routing_repair_update_search import (
    RoutingRepairUpdateSearchContext,
    apply_routing_repair_update_search,
)


def run_direct_answer_training_loop(
    *,
    args: Any,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    direct_lessons: dict[Any, Any],
    direct_training_pool: list[Any],
    direct_training_cursor: Any,
    direct_rng: Any,
    direct_steps_to_run: int,
    direct_answer_terminator: str,
    direct_params: Any,
    direct_answer_baseline_floor_update_gate_active: bool,
    direct_answer_baseline_floor_adaptive_updates_active: bool,
    direct_answer_update_guard: dict[str, Any],
    direct_baseline: dict[str, Any],
    direct_snapshot_recorder: Any,
    best_direct_snapshot: Any,
    last_direct_snapshot: dict[str, Any],
    last_direct_snapshot_step: int,
    train_adaptive_baseline_floor_update: Callable[..., float],
    train_baseline_anchored_prompt: Callable[..., float],
    restore_direct_update_state: Callable[[dict[str, Any], dict[str, Any]], None],
    train_mode_step: Callable[..., Any] = train_direct_answer_mode_step,
    apply_guard_probe: Callable[..., None] = apply_direct_update_guard_probe,
) -> DirectAnswerLoopResult:
    running_direct_loss = 0.0
    routing_repair_batch_steps: list[dict[str, Any]] = []
    routing_repair_guard_active = routing_repair_batch_evidence_enabled(args)
    frontier_update_guard_active = (
        direct_answer_frontier_update_guard_active(args)
        and not direct_answer_baseline_floor_update_gate_active
        and not routing_repair_guard_active
    )
    update_guard_probe_active = (
        direct_answer_baseline_floor_update_gate_active
        or routing_repair_guard_active
        or frontier_update_guard_active
    )
    current_state = {
        "model": model,
        "tokenizer": tokenizer,
        "optimizer": optimizer,
        "params": direct_params,
    }

    def restore_and_refresh_state(
        model_payload: dict[str, Any],
        optimizer_payload: dict[str, Any],
    ) -> None:
        restored = restore_direct_update_state(model_payload, optimizer_payload)
        if _is_restored_state_tuple(restored):
            (
                current_state["model"],
                current_state["tokenizer"],
                current_state["optimizer"],
                current_state["params"],
            ) = restored

    for direct_step in range(1, direct_steps_to_run + 1):
        example = direct_training_cursor.next()
        pre_update_model_payload: dict[str, Any] | None = None
        pre_update_optimizer_payload: dict[str, Any] | None = None
        pre_update_rng_state = None
        if update_guard_probe_active:
            pre_update_model_payload = current_state["model"].to_dict(
                current_state["tokenizer"]
            )
            pre_update_optimizer_payload = current_state["optimizer"].to_dict()
            pre_update_rng_state = direct_rng.getstate()
        routing_repair_batch_step = record_routing_repair_batch_step(
            args=args,
            model=current_state["model"],
            tokenizer=current_state["tokenizer"],
            branch_examples=direct_training_pool,
            rng=direct_rng,
            direct_step=direct_step,
            terminator=direct_answer_terminator,
        )
        if routing_repair_batch_step is not None:
            routing_repair_batch_steps.append(routing_repair_batch_step)

        train_adaptive_update = (
            lambda step, model_payload, optimizer_payload, rng_state: (
                train_adaptive_baseline_floor_update(
                    example,
                    step,
                    model_payload,
                    optimizer_payload,
                    rng_state,
                )
            )
        )
        if routing_repair_guard_active and _has_update_payloads(
            pre_update_model_payload,
            pre_update_optimizer_payload,
        ):
            mode_step_result = apply_routing_repair_update_search(
                RoutingRepairUpdateSearchContext(
                    args=args,
                    direct_step=direct_step,
                    example=example,
                    lesson=direct_lessons[example],
                    branch_examples=direct_training_pool,
                    rng=direct_rng,
                    terminator=direct_answer_terminator,
                    direct_baseline=direct_baseline,
                    direct_snapshot_recorder=direct_snapshot_recorder,
                    direct_answer_update_guard=direct_answer_update_guard,
                    model=lambda: current_state["model"],
                    tokenizer=lambda: current_state["tokenizer"],
                    optimizer=lambda: current_state["optimizer"],
                    params=lambda: current_state["params"],
                    restore_state=restore_and_refresh_state,
                    train_mode_step=train_mode_step,
                    train_adaptive_baseline_floor_update=train_adaptive_update,
                    train_baseline_anchored_prompt=train_baseline_anchored_prompt,
                    pre_update_model_payload=pre_update_model_payload,
                    pre_update_optimizer_payload=pre_update_optimizer_payload,
                    pre_update_rng_state=pre_update_rng_state,
                )
            )
        else:
            mode_step_result = train_mode_step(
                args=args,
                model=current_state["model"],
                tokenizer=current_state["tokenizer"],
                example=example,
                lesson=direct_lessons[example],
                branch_examples=direct_training_pool,
                rng=direct_rng,
                direct_step=direct_step,
                terminator=direct_answer_terminator,
                params=current_state["params"],
                baseline_floor_adaptive_updates_active=(
                    direct_answer_baseline_floor_adaptive_updates_active
                ),
                pre_update_model_payload=pre_update_model_payload,
                pre_update_optimizer_payload=pre_update_optimizer_payload,
                pre_update_rng_state=pre_update_rng_state,
                train_adaptive_baseline_floor_update=train_adaptive_update,
                train_baseline_anchored_prompt=train_baseline_anchored_prompt,
            )
        running_direct_loss += mode_step_result.loss
        if update_guard_probe_active and not mode_step_result.update_guard_applied:
            if frontier_update_guard_active:
                apply_direct_frontier_update_guard_probe(
                    direct_answer_update_guard=direct_answer_update_guard,
                    direct_baseline=direct_baseline,
                    direct_step=direct_step,
                    direct_snapshot_recorder=direct_snapshot_recorder,
                    frontier_metrics_path=getattr(
                        args,
                        "direct_answer_frontier_metrics",
                        None,
                    ),
                    pre_update_model_payload=pre_update_model_payload,
                    pre_update_optimizer_payload=pre_update_optimizer_payload,
                    restore_direct_update_state=restore_and_refresh_state,
                )
            else:
                apply_guard_probe(
                    direct_answer_update_guard=direct_answer_update_guard,
                    direct_baseline=direct_baseline,
                    direct_step=direct_step,
                    direct_snapshot_recorder=direct_snapshot_recorder,
                    pre_update_model_payload=pre_update_model_payload,
                    pre_update_optimizer_payload=pre_update_optimizer_payload,
                    restore_direct_update_state=restore_and_refresh_state,
                )
        if args.direct_answer_eval_every > 0 and (
            direct_step % args.direct_answer_eval_every == 0
        ):
            train_loss = running_direct_loss / args.direct_answer_eval_every
            last_direct_snapshot = direct_snapshot_recorder.append(
                direct_step,
                train_loss,
            )
            last_direct_snapshot_step = direct_step
            best_direct_snapshot.record(
                last_direct_snapshot,
                current_state["model"],
                current_state["tokenizer"],
                current_state["optimizer"],
            )
            print(f"direct_answer_step={direct_step} train_loss={train_loss:.4f}")
            running_direct_loss = 0.0
    return DirectAnswerLoopResult(
        last_snapshot=last_direct_snapshot,
        last_snapshot_step=last_direct_snapshot_step,
        routing_repair_batch_evidence=routing_repair_batch_evidence_summary(
            args,
            routing_repair_batch_steps,
            direct_baseline,
        ),
    )


def _has_update_payloads(
    model_payload: dict[str, Any] | None,
    optimizer_payload: dict[str, Any] | None,
) -> bool:
    return model_payload is not None and optimizer_payload is not None


def _is_restored_state_tuple(value: Any) -> bool:
    return isinstance(value, tuple) and len(value) == 4
