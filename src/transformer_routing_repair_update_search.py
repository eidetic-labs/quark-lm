"""Coverage-preserving update search for routing-repair bundles."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from branch_diversity_snapshot_coverage import (
    branch_diversity_snapshot_preserves_target_coverage,
    branch_diversity_snapshot_target_coverage_delta,
)
from branch_diversity_snapshot_stability import (
    branch_diversity_snapshot_preserves_profile_stability,
)
from branch_diversity_snapshots import branch_diversity_snapshot_score_improved
from transformer_direct_answer_mode_dispatch import DirectAnswerModeStepResult
from transformer_direct_answer_update_guard import (
    record_direct_update_guard_acceptance,
    record_direct_update_guard_rejection_attempt,
)
from transformer_routing_repair_optimizer_diagnostics import (
    record_routing_repair_optimizer_probe,
)
from transformer_routing_repair_neutral_updates import (
    ROUTING_REPAIR_NEUTRAL_UPDATE_SHAPE,
    record_routing_repair_neutral_acceptance,
)

ROUTING_REPAIR_LEARNING_RATE_SCALES = (1.0, 0.5, 0.25, 0.125, 0.0625)
ROUTING_REPAIR_UPDATE_SHAPE = "routing_repair_scaled"


@dataclass(frozen=True)
class RoutingRepairUpdateSearchContext:
    args: Any
    direct_step: int
    example: Any
    lesson: Any
    branch_examples: list[Any]
    eval_records: dict[str, list[dict[str, Any]]] | None
    rng: Any
    terminator: str
    direct_baseline: dict[str, Any]
    direct_snapshot_recorder: Any
    direct_answer_update_guard: dict[str, Any]
    model: Callable[[], Any]
    tokenizer: Callable[[], Any]
    optimizer: Callable[[], Any]
    params: Callable[[], Any]
    restore_state: Callable[[dict[str, Any], dict[str, Any]], None]
    train_mode_step: Callable[..., Any]
    train_adaptive_baseline_floor_update: Callable[..., float]
    train_baseline_anchored_prompt: Callable[..., float]
    pre_update_model_payload: dict[str, Any]
    pre_update_optimizer_payload: dict[str, Any]
    pre_update_rng_state: object | None


def apply_routing_repair_update_search(
    ctx: RoutingRepairUpdateSearchContext,
) -> DirectAnswerModeStepResult:
    """Try bounded learning-rate scales and keep the first coverage-safe update."""

    guard = ctx.direct_answer_update_guard
    guard["routing_repair_scaled_retry_active"] = True
    guard["routing_repair_learning_rate_scales"] = list(
        ROUTING_REPAIR_LEARNING_RATE_SCALES
    )
    guard["checked_steps"] += 1
    last_loss = 0.0
    for learning_rate_scale in ROUTING_REPAIR_LEARNING_RATE_SCALES:
        ctx.restore_state(
            ctx.pre_update_model_payload,
            ctx.pre_update_optimizer_payload,
        )
        _restore_rng(ctx.rng, ctx.pre_update_rng_state)
        guard["attempted_updates"] += 1
        result = _train_scaled_update(ctx, learning_rate_scale)
        last_loss = float(result.loss)
        record_routing_repair_optimizer_probe(
            guard,
            ctx.optimizer(),
            learning_rate_scale,
            last_loss,
        )
        probe_snapshot = _probe_snapshot(ctx, learning_rate_scale)
        coverage_preserved = branch_diversity_snapshot_preserves_target_coverage(
            probe_snapshot,
            ctx.direct_baseline,
        )
        stability_preserved = branch_diversity_snapshot_preserves_profile_stability(
            probe_snapshot,
            ctx.direct_baseline,
        )
        branch_response = _branch_response_recorded(
            probe_snapshot,
            ctx.direct_baseline,
        )
        if coverage_preserved and stability_preserved and branch_response:
            _record_branch_response(guard, learning_rate_scale)
            record_direct_update_guard_acceptance(
                guard,
                learning_rate_scale,
                ROUTING_REPAIR_UPDATE_SHAPE,
            )
            guard["routing_repair_accepted_learning_rate_scale"] = (
                learning_rate_scale
            )
            return DirectAnswerModeStepResult(last_loss, update_guard_applied=True)
        if coverage_preserved and stability_preserved:
            record_routing_repair_neutral_acceptance(guard, learning_rate_scale)
            record_direct_update_guard_acceptance(
                guard,
                learning_rate_scale,
                ROUTING_REPAIR_NEUTRAL_UPDATE_SHAPE,
            )
            guard["routing_repair_accepted_learning_rate_scale"] = (
                learning_rate_scale
            )
            return DirectAnswerModeStepResult(last_loss, update_guard_applied=True)
        if coverage_preserved and not stability_preserved:
            _record_stability_rejection(guard, learning_rate_scale)
        record_direct_update_guard_rejection_attempt(
            guard,
            ctx.direct_baseline,
            ctx.direct_step,
            probe_snapshot,
            learning_rate_scale,
            ROUTING_REPAIR_UPDATE_SHAPE,
        )
    guard["rejected_steps"] += 1
    ctx.restore_state(
        ctx.pre_update_model_payload,
        ctx.pre_update_optimizer_payload,
    )
    _restore_rng(ctx.rng, ctx.pre_update_rng_state)
    guard["routing_repair_accepted_learning_rate_scale"] = None
    return DirectAnswerModeStepResult(last_loss, update_guard_applied=True)


def _train_scaled_update(
    ctx: RoutingRepairUpdateSearchContext,
    learning_rate_scale: float,
) -> Any:
    scaled_args = copy.copy(ctx.args)
    scaled_args.direct_answer_learning_rate = (
        float(ctx.args.direct_answer_learning_rate) * learning_rate_scale
    )
    return ctx.train_mode_step(
        args=scaled_args,
        model=ctx.model(),
        tokenizer=ctx.tokenizer(),
        example=ctx.example,
        lesson=ctx.lesson,
        branch_examples=ctx.branch_examples,
        eval_records=ctx.eval_records,
        rng=ctx.rng,
        direct_step=ctx.direct_step,
        terminator=ctx.terminator,
        params=ctx.params(),
        baseline_floor_adaptive_updates_active=False,
        pre_update_model_payload=ctx.pre_update_model_payload,
        pre_update_optimizer_payload=ctx.pre_update_optimizer_payload,
        pre_update_rng_state=ctx.pre_update_rng_state,
        train_adaptive_baseline_floor_update=(
            ctx.train_adaptive_baseline_floor_update
        ),
        train_baseline_anchored_prompt=ctx.train_baseline_anchored_prompt,
    )


def _probe_snapshot(
    ctx: RoutingRepairUpdateSearchContext,
    learning_rate_scale: float,
) -> dict[str, Any]:
    return ctx.direct_snapshot_recorder.record(
        ctx.direct_step,
        None,
        {
            "routing_repair_update_search_probe": True,
            "learning_rate_scale": learning_rate_scale,
        },
    )


def _branch_response_recorded(
    snapshot: dict[str, Any],
    baseline: dict[str, Any],
) -> bool:
    diversity = snapshot.get("branch_diversity_target", {})
    if isinstance(diversity, dict) and diversity.get("passed") is True:
        return True
    delta = branch_diversity_snapshot_target_coverage_delta(snapshot, baseline)
    return (
        int(delta.get("improved_profile_count", 0)) > 0
        or branch_diversity_snapshot_score_improved(snapshot, baseline)
    )


def _record_branch_response(
    guard: dict[str, Any],
    learning_rate_scale: float,
) -> None:
    guard["routing_repair_branch_response_acceptances"] = int(
        guard.get("routing_repair_branch_response_acceptances", 0)
    ) + 1
    scales = guard.setdefault("routing_repair_branch_response_learning_rate_scales", [])
    if isinstance(scales, list):
        scales.append(learning_rate_scale)


def _record_stability_rejection(
    guard: dict[str, Any],
    learning_rate_scale: float,
) -> None:
    guard["routing_repair_stability_rejections"] = int(
        guard.get("routing_repair_stability_rejections", 0)
    ) + 1
    scales = guard.setdefault("routing_repair_stability_learning_rate_scales", [])
    if isinstance(scales, list):
        scales.append(learning_rate_scale)


def _restore_rng(rng: Any, rng_state: object | None) -> None:
    if rng_state is not None and hasattr(rng, "setstate"):
        rng.setstate(rng_state)
