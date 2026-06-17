"""Baseline-floor stabilization stage routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from transformer_baseline_floor_training import (
    train_direct_answer_baseline_floor_stabilization_batch_stage,
)
from transformer_baseline_floor_profile_scale_stage import (
    train_baseline_floor_profile_scale_stage,
)
from transformer_baseline_floor_sequential import (
    train_baseline_floor_sequential_profile_stage,
)
from transformer_direct_modes import BASELINE_FLOOR_STABILIZATION_ANCHOR_BATCH_SIZE


@dataclass(frozen=True)
class BaselineFloorStabilizationContext:
    args: Any
    direct_setup: Any
    model: Callable[[], Any]
    tokenizer: Callable[[], Any]
    optimizer: Callable[[], Any]
    rng: Any
    params: Callable[[], Any]
    update_guard: dict[str, Any]
    direct_baseline: dict[str, Any]
    snapshot_recorder: Any
    restore_direct_update_state: Callable[[dict[str, Any], dict[str, Any]], None]


def train_baseline_floor_stabilization_stage(
    ctx: BaselineFloorStabilizationContext,
    update_learning_rate: float,
    direct_step: int,
    *,
    profile_scale_stage: Callable[..., tuple[float, bool]] = (
        train_baseline_floor_profile_scale_stage
    ),
    sequential_stage: Callable[..., tuple[float, bool]] = (
        train_baseline_floor_sequential_profile_stage
    ),
    batch_stage: Callable[..., tuple[float, bool]] = (
        train_direct_answer_baseline_floor_stabilization_batch_stage
    ),
) -> tuple[float, bool]:
    setup = ctx.direct_setup
    repair_anchors = setup.direct_baseline_floor_repair_anchors
    if not repair_anchors:
        return 0.0, False
    if setup.direct_answer_baseline_floor_profile_scale_calibrated_stabilization_active:
        return profile_scale_stage(ctx, direct_step)
    if setup.direct_answer_baseline_floor_sequential_stabilization_active:
        return sequential_stage(
            model=ctx.model(),
            tokenizer=ctx.tokenizer(),
            optimizer=ctx.optimizer(),
            repair_anchors=repair_anchors,
            rng=ctx.rng,
            update_learning_rate=update_learning_rate,
            base_learning_rate=ctx.args.direct_answer_learning_rate,
            update_guard=ctx.update_guard,
            direct_step=direct_step,
            direct_baseline=ctx.direct_baseline,
            snapshot_recorder=ctx.snapshot_recorder,
            restore_direct_update_state=ctx.restore_direct_update_state,
            calibrated=(
                setup.direct_answer_baseline_floor_calibrated_sequential_stabilization_active
            ),
            params=ctx.params(),
        )
    stabilization_batch_size = (
        len(repair_anchors)
        if setup.direct_answer_baseline_floor_profile_targeted_stabilization_active
        else BASELINE_FLOOR_STABILIZATION_ANCHOR_BATCH_SIZE
    )
    return batch_stage(
        ctx.model(),
        repair_anchors,
        ctx.rng,
        update_learning_rate,
        stabilization_batch_size,
        ctx.update_guard,
        params=ctx.params(),
    )
