"""Baseline-floor sequential profile stabilization stage."""

from __future__ import annotations

import random
from typing import Any, Callable

from branch_diversity_snapshots import (
    branch_diversity_snapshot_preserves_target_coverage,
    branch_diversity_snapshot_target_coverage_diagnostics,
)
from replay_plan import BranchReplayRecord
from transformer_baseline_floor_anchor_profiles import (
    baseline_floor_anchor_profile_groups,
)
from transformer_baseline_floor_anchor_selection import (
    baseline_floor_objective_anchor_batch,
)
from transformer_baseline_floor_training import (
    train_direct_answer_baseline_floor_anchor_batch,
)
from transformer_baseline_floor_sequential_samples import (
    record_baseline_floor_sequential_profile_probe_result,
)


def train_baseline_floor_sequential_profile_stage(
    *,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    repair_anchors: list[BranchReplayRecord],
    rng: random.Random,
    update_learning_rate: float,
    base_learning_rate: float,
    update_guard: dict[str, Any],
    direct_step: int,
    direct_baseline: dict[str, Any],
    snapshot_recorder: Any,
    restore_direct_update_state: Callable[[dict[str, Any], dict[str, Any]], None],
    calibrated: bool,
    params: Any = None,
    batch_factory: Callable[
        [list[BranchReplayRecord], random.Random, int],
        list[BranchReplayRecord],
    ] = baseline_floor_objective_anchor_batch,
    train_anchor_batch: Callable[..., float] = (
        train_direct_answer_baseline_floor_anchor_batch
    ),
    preserves_target_coverage: Callable[
        [dict[str, Any], dict[str, Any]], bool
    ] = branch_diversity_snapshot_preserves_target_coverage,
    coverage_diagnostics: Callable[
        [dict[str, Any], dict[str, Any]], dict[str, Any]
    ] = branch_diversity_snapshot_target_coverage_diagnostics,
) -> tuple[float, bool]:
    profile_groups = baseline_floor_anchor_profile_groups(repair_anchors)
    sequential_update_shape = (
        "calibrated_sequential_profile_stabilization"
        if calibrated
        else "sequential_profile_stabilization"
    )
    total_loss = 0.0
    accepted_any = False
    for profile, profile_anchors in profile_groups.items():
        profile_model_payload = model.to_dict(tokenizer)
        profile_optimizer_payload = optimizer.to_dict()
        profile_rng_state = rng.getstate()
        profile_batch = batch_factory(
            profile_anchors,
            rng,
            len(profile_anchors),
        )
        _record_sequential_profile_attempt(
            update_guard,
            len(profile_batch),
        )
        profile_loss = train_anchor_batch(
            model,
            profile_batch,
            update_learning_rate,
            params=params,
        )
        total_loss += profile_loss
        profile_probe_snapshot = snapshot_recorder.record(
            direct_step,
            None,
            {
                "baseline_floor_update_guard_probe": True,
                "baseline_floor_sequential_profile_probe": True,
                "baseline_floor_calibrated_sequential_profile_probe": calibrated,
                "learning_rate_scale": (
                    update_learning_rate / max(base_learning_rate, 1e-12)
                ),
                "update_shape": sequential_update_shape,
                "sequential_profile": profile,
                "sequential_profile_records": len(profile_batch),
            },
        )
        profile_floor_preserved = preserves_target_coverage(
            profile_probe_snapshot,
            direct_baseline,
        )
        diagnostics = None
        if profile_floor_preserved:
            accepted_any = True
        else:
            diagnostics = coverage_diagnostics(
                profile_probe_snapshot,
                direct_baseline,
            )
        record_baseline_floor_sequential_profile_probe_result(
            update_guard,
            profile=profile,
            accepted=profile_floor_preserved,
            records=len(profile_batch),
            diagnostics=diagnostics,
        )
        if not profile_floor_preserved:
            restore_direct_update_state(
                profile_model_payload,
                profile_optimizer_payload,
            )
            rng.setstate(profile_rng_state)
    return total_loss / max(len(profile_groups), 1), accepted_any


def _record_sequential_profile_attempt(
    update_guard: dict[str, Any],
    records: int,
) -> None:
    update_guard["sequential_profile_attempts"] += 1
    update_guard["sequential_profile_records"] += records
    update_guard["stabilization_anchor_batches"] += 1
    update_guard["stabilization_anchor_records"] += records
