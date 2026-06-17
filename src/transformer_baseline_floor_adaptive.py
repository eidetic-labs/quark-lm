"""Baseline-floor adaptive update gate."""

from __future__ import annotations

import random
from typing import Any, Callable, Sequence

from branch_diversity_snapshots import (
    branch_diversity_snapshot_preserves_target_coverage,
    branch_diversity_snapshot_score,
)
from replay_plan import BranchReplayRecord
from transformer_baseline_floor_training import (
    train_direct_answer_baseline_floor_anchor_repair_stage,
)
from transformer_baseline_floor_update_shapes import baseline_floor_attempt_update_shape
from transformer_direct_answer_update_guard import (
    record_direct_update_guard_acceptance,
    record_direct_update_guard_rejection_attempt,
)


def train_adaptive_baseline_floor_update_stage(
    *,
    model: Any,
    rng: random.Random,
    example: Any,
    lesson: Any,
    direct_step: int,
    base_model_payload: dict[str, Any],
    base_optimizer_payload: dict[str, Any],
    base_rng_state: object,
    direct_answer_mode: str,
    direct_answer_learning_rate: float,
    direct_answer_branch_batch_size: int,
    direct_answer_hard_negatives: int,
    update_guard: dict[str, Any],
    direct_baseline: dict[str, Any],
    snapshot_recorder: Any,
    outer_learning_rate_scales: Sequence[float],
    repair_anchors: list[BranchReplayRecord],
    repaired_updates_active: bool,
    stabilization_active: bool,
    profile_scale_diversity_active: bool,
    train_stabilization_update: Callable[[float, int], tuple[float, bool]],
    train_baseline_anchored_prompt: Callable[[Any, Any, float], float],
    restore_direct_update_state: Callable[[dict[str, Any], dict[str, Any]], None],
    params: Any = None,
    train_repair_stage: Callable[..., float] = (
        train_direct_answer_baseline_floor_anchor_repair_stage
    ),
    preserves_target_coverage: Callable[
        [dict[str, Any], dict[str, Any]], bool
    ] = branch_diversity_snapshot_preserves_target_coverage,
    snapshot_score: Callable[[dict[str, Any]], tuple[float, ...]] = (
        branch_diversity_snapshot_score
    ),
    update_shape_for_mode: Callable[[str], str] = baseline_floor_attempt_update_shape,
    record_acceptance: Callable[[dict[str, Any], float, str], None] = (
        record_direct_update_guard_acceptance
    ),
    record_rejection_attempt: Callable[
        [dict[str, Any], dict[str, Any], int, dict[str, Any], float, str],
        None,
    ] = record_direct_update_guard_rejection_attempt,
) -> float:
    last_loss = 0.0
    update_guard["checked_steps"] += 1
    for learning_rate_scale in outer_learning_rate_scales:
        restore_direct_update_state(base_model_payload, base_optimizer_payload)
        rng.setstate(base_rng_state)
        update_guard["attempted_updates"] += 1
        attempt_update_shape = update_shape_for_mode(direct_answer_mode)
        update_learning_rate = direct_answer_learning_rate * learning_rate_scale
        update_applied = True
        if stabilization_active:
            last_loss, update_applied = train_stabilization_update(
                update_learning_rate,
                direct_step,
            )
        else:
            last_loss = train_baseline_anchored_prompt(
                example,
                lesson,
                update_learning_rate,
            )
        probe_snapshot = snapshot_recorder.record(
            direct_step,
            None,
            {
                "baseline_floor_update_guard_probe": True,
                "learning_rate_scale": learning_rate_scale,
                "update_shape": attempt_update_shape,
            },
        )
        if update_applied and preserves_target_coverage(
            probe_snapshot,
            direct_baseline,
        ):
            if (
                profile_scale_diversity_active
                and snapshot_score(probe_snapshot) < snapshot_score(direct_baseline)
            ):
                update_guard["profile_scale_diversity_outer_rejections"] += 1
            else:
                if profile_scale_diversity_active:
                    update_guard["profile_scale_diversity_outer_acceptances"] += 1
                record_acceptance(
                    update_guard,
                    learning_rate_scale,
                    attempt_update_shape,
                )
                return last_loss
        if not update_applied:
            update_guard["rejected_no_effective_update_attempts"] += 1
        rejection_snapshot = probe_snapshot
        rejection_update_shape = attempt_update_shape
        if repaired_updates_active and repair_anchors:
            update_guard["repair_attempts"] += 1
            repair_loss = train_repair_stage(
                model,
                repair_anchors,
                rng,
                update_learning_rate,
                direct_answer_branch_batch_size,
                direct_answer_hard_negatives,
                update_guard,
                params=params,
            )
            repaired_probe_snapshot = snapshot_recorder.record(
                direct_step,
                None,
                {
                    "baseline_floor_update_guard_probe": True,
                    "baseline_floor_repair_probe": True,
                    "learning_rate_scale": learning_rate_scale,
                },
            )
            if preserves_target_coverage(
                repaired_probe_snapshot,
                direct_baseline,
            ):
                record_acceptance(update_guard, learning_rate_scale, "repaired")
                return (last_loss + repair_loss) / 2.0
            rejection_snapshot = repaired_probe_snapshot
            rejection_update_shape = "repaired"
        record_rejection_attempt(
            update_guard,
            direct_baseline,
            direct_step,
            rejection_snapshot,
            learning_rate_scale,
            rejection_update_shape,
        )
    update_guard["rejected_steps"] += 1
    restore_direct_update_state(base_model_payload, base_optimizer_payload)
    rng.setstate(base_rng_state)
    return last_loss
