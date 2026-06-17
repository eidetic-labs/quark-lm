"""Optional profile-scale recovery attempts for baseline-floor stabilization."""

from __future__ import annotations

from typing import Any

from transformer_baseline_floor_attempt_state import (
    BaselineFloorProfileAttemptState,
)
from transformer_baseline_floor_owner_preservation import (
    check_owner_paraphrase_binding_preservation,
)
from transformer_baseline_floor_coverage_recovery import (
    try_baseline_floor_coverage_recovery,
)
from transformer_baseline_floor_profile_scale_recovery_attempts import (
    try_profile_scale_branch_diversity_recovery,
    try_profile_scale_collapsed_profile_binding,
    try_profile_scale_missing_first_token,
)
from transformer_direct_modes import (
    BASELINE_FLOOR_OWNER_PARAPHRASE_BINDING_PRESERVED_PROFILES,
)


def apply_profile_scale_recovery_attempts(
    ctx: Any,
    *,
    state: BaselineFloorProfileAttemptState,
    profile: str,
    profile_batch: list[Any],
    frontier_targets_by_profile: dict[str, set[int]],
    frontier_records: int,
    priorities: dict[str, bool],
    profile_base_snapshot: dict[str, Any] | None,
    profile_base_score: tuple[float, ...] | None,
    profile_scale: float,
    update_shape: str,
    direct_step: int,
) -> tuple[float, int, bool, bool]:
    total_loss = 0.0
    loss_count = 0
    coverage = _try_coverage_recovery(
        ctx,
        state,
        profile,
        profile_batch,
        frontier_targets_by_profile,
        frontier_records,
        profile_base_snapshot,
        profile_base_score,
        profile_scale,
        update_shape,
        direct_step,
    )
    total_loss += coverage.loss_total
    loss_count += coverage.loss_count
    state.apply_coverage_recovery(coverage)
    diversity_ok = state.diversity_accepted(
        ctx.direct_setup.direct_answer_baseline_floor_profile_scale_diversity_stabilization_active
    )
    preservation = check_owner_paraphrase_binding_preservation(
        active=(
            ctx.direct_setup.direct_answer_baseline_floor_profile_scale_owner_paraphrase_binding_frontier_stabilization_active
        ),
        update_guard=ctx.update_guard,
        profile_probe_snapshot=state.profile_probe_snapshot,
        profile_base_snapshot=profile_base_snapshot,
        preserved_profiles=BASELINE_FLOOR_OWNER_PARAPHRASE_BINDING_PRESERVED_PROFILES,
    )
    state.apply_owner_paraphrase_preservation(preservation)
    diversity_ok = diversity_ok and preservation.preserved
    state.enforce_coverage_tie_requirement(
        ctx.direct_setup.direct_answer_baseline_floor_profile_scale_coverage_prep_frontier_stabilization_active
    )
    coverage_ok = state.coverage_accepted(
        ctx.direct_setup.direct_answer_baseline_floor_profile_scale_coverage_frontier_stabilization_active
    )
    for result in (
        try_profile_scale_branch_diversity_recovery(
            ctx,
            state,
            profile,
            profile_batch,
            frontier_records,
            diversity_ok,
            coverage_ok,
            profile_scale,
            update_shape,
            direct_step,
        ),
        try_profile_scale_collapsed_profile_binding(
            ctx,
            state,
            profile,
            profile_batch,
            frontier_records,
            diversity_ok,
            coverage_ok,
            profile_scale,
            update_shape,
            direct_step,
        ),
        try_profile_scale_missing_first_token(
            ctx,
            state,
            profile,
            profile_batch,
            frontier_records,
            priorities,
            profile_base_snapshot,
            diversity_ok,
            coverage_ok,
            profile_scale,
            update_shape,
            direct_step,
        ),
    ):
        total_loss += result.loss_total
        loss_count += result.loss_count
    return total_loss, loss_count, diversity_ok, coverage_ok


def _try_coverage_recovery(
    ctx: Any,
    state: BaselineFloorProfileAttemptState,
    profile: str,
    profile_batch: list[Any],
    frontier_targets_by_profile: dict[str, set[int]],
    frontier_records: int,
    profile_base_snapshot: dict[str, Any] | None,
    profile_base_score: tuple[float, ...] | None,
    profile_scale: float,
    update_shape: str,
    direct_step: int,
) -> Any:
    setup = ctx.direct_setup
    return try_baseline_floor_coverage_recovery(
        active=setup.direct_answer_baseline_floor_profile_scale_coverage_recovery_frontier_stabilization_active,
        coverage_prep_accepted=state.coverage_prep_accepted,
        profile_base_snapshot=profile_base_snapshot,
        profile_base_score=profile_base_score,
        profile_score=state.profile_score,
        profile_probe_snapshot=state.profile_probe_snapshot,
        coverage_delta=state.coverage_delta,
        coverage_outcome=state.coverage_outcome,
        coverage_rejection_reason=state.coverage_rejection_reason,
        floor_preserved=state.floor_preserved,
        diversity_outcome=state.diversity_outcome,
        diversity_rejection_reason=state.diversity_rejection_reason,
        branch_stable_active=setup.direct_answer_baseline_floor_profile_scale_branch_stable_coverage_recovery_frontier_stabilization_active,
        model=ctx.model(),
        tokenizer=ctx.tokenizer(),
        optimizer=ctx.optimizer(),
        profile_batch=profile_batch,
        frontier_targets_by_profile=frontier_targets_by_profile,
        base_learning_rate=ctx.args.direct_answer_learning_rate,
        profile_scale=profile_scale,
        params=ctx.params(),
        direct_step=direct_step,
        direct_baseline=ctx.direct_baseline,
        snapshot_recorder=ctx.snapshot_recorder,
        update_guard=ctx.update_guard,
        update_shape=update_shape,
        profile=profile,
        profile_frontier_records=frontier_records,
        restore_direct_update_state=ctx.restore_direct_update_state,
    )
