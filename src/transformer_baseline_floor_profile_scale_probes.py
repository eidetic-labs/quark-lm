"""Profile-scale baseline-floor probes and attempt state construction."""

from __future__ import annotations

from typing import Any

from branch_diversity_snapshots import branch_diversity_snapshot_score
from transformer_baseline_floor_attempt_state import (
    BaselineFloorProfileAttemptState,
)
from transformer_baseline_floor_profile_outcomes import (
    evaluate_baseline_floor_profile_outcome,
)


def record_baseline_floor_profile_base_probe(
    ctx: Any,
    profile: str,
    update_shape: str,
    direct_step: int,
) -> tuple[dict[str, Any] | None, tuple[float, ...] | None]:
    if not (
        ctx.direct_setup.direct_answer_baseline_floor_profile_scale_diversity_stabilization_active
    ):
        return None, None
    snapshot = ctx.snapshot_recorder.record(
        direct_step,
        None,
        {
            "baseline_floor_update_guard_probe": True,
            "baseline_floor_profile_scale_diversity_base_probe": True,
            "update_shape": update_shape,
            "sequential_profile": profile,
        },
    )
    return snapshot, branch_diversity_snapshot_score(snapshot)


def record_baseline_floor_profile_scale_probe(
    ctx: Any,
    *,
    profile: str,
    profile_batch: list[Any],
    frontier_records: int,
    profile_scale: float,
    update_shape: str,
    priorities: dict[str, bool],
    direct_step: int,
) -> dict[str, Any]:
    return ctx.snapshot_recorder.record(
        direct_step,
        None,
        {
            "baseline_floor_update_guard_probe": True,
            "baseline_floor_sequential_profile_probe": True,
            "baseline_floor_calibrated_sequential_profile_probe": True,
            "baseline_floor_profile_scale_memory_probe": True,
            "baseline_floor_profile_scale_frontier_probe": (
                ctx.direct_setup.direct_answer_baseline_floor_profile_scale_frontier_stabilization_active
            ),
            "learning_rate_scale": profile_scale,
            "update_shape": update_shape,
            "sequential_profile": profile,
            "sequential_profile_records": len(profile_batch),
            "sequential_profile_frontier_records": frontier_records,
            "remaining_profile_binding_prioritized": priorities["remaining"],
            "owner_paraphrase_binding_prioritized": priorities["owner"],
            "memory_consolidation_prioritized": priorities["memory"],
            "memory_consolidation_target_profiles": (
                ctx.direct_setup.direct_memory_consolidation_target_profiles
            ),
        },
    )


def initial_baseline_floor_profile_attempt_state(
    ctx: Any,
    *,
    snapshot: dict[str, Any],
    profile_base_snapshot: dict[str, Any] | None,
    profile_base_score: tuple[float, ...] | None,
) -> BaselineFloorProfileAttemptState:
    setup = ctx.direct_setup
    outcome = evaluate_baseline_floor_profile_outcome(
        profile_probe_snapshot=snapshot,
        direct_baseline=ctx.direct_baseline,
        profile_base_snapshot=profile_base_snapshot,
        profile_base_score=profile_base_score,
        diversity_active=(
            setup.direct_answer_baseline_floor_profile_scale_diversity_stabilization_active
        ),
        coverage_frontier_active=(
            setup.direct_answer_baseline_floor_profile_scale_coverage_frontier_stabilization_active
        ),
        coverage_prep_frontier_active=(
            setup.direct_answer_baseline_floor_profile_scale_coverage_prep_frontier_stabilization_active
        ),
    )
    return BaselineFloorProfileAttemptState.from_profile_outcome(
        outcome,
        snapshot,
        profile_base_score,
    )
