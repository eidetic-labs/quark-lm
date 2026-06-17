"""Specialized profile-scale recovery attempt adapters."""

from __future__ import annotations

from typing import Any

from transformer_baseline_floor_attempt_state import (
    BaselineFloorProfileAttemptState,
)
from transformer_baseline_floor_binding import (
    try_baseline_floor_collapsed_profile_binding,
)
from transformer_baseline_floor_branch_diversity_recovery import (
    try_baseline_floor_branch_diversity_recovery,
)
from transformer_baseline_floor_memory import (
    try_baseline_floor_missing_first_token_consolidation,
)
from transformer_direct_modes import (
    BASELINE_FLOOR_OWNER_PARAPHRASE_BINDING_PRESERVED_PROFILES,
    BASELINE_FLOOR_OWNER_PARAPHRASE_BINDING_TARGET_PROFILES,
)


def try_profile_scale_branch_diversity_recovery(
    ctx: Any,
    state: BaselineFloorProfileAttemptState,
    profile: str,
    profile_batch: list[Any],
    frontier_records: int,
    diversity_ok: bool,
    coverage_ok: bool,
    profile_scale: float,
    update_shape: str,
    direct_step: int,
) -> Any:
    setup = ctx.direct_setup
    result = try_baseline_floor_branch_diversity_recovery(
        active=setup.direct_answer_baseline_floor_profile_scale_branch_diversity_recovery_frontier_stabilization_active,
        floor_preserved=state.floor_preserved,
        diversity_accepted=diversity_ok,
        coverage_accepted=coverage_ok,
        profile_score=state.profile_score,
        profile_probe_snapshot=state.profile_probe_snapshot,
        model=ctx.model(),
        tokenizer=ctx.tokenizer(),
        optimizer=ctx.optimizer(),
        profile_batch=profile_batch,
        base_learning_rate=ctx.args.direct_answer_learning_rate,
        profile_scale=profile_scale,
        negative_weight=ctx.args.direct_answer_negative_weight,
        positive_weight=ctx.args.direct_answer_positive_weight,
        contrast_weight=ctx.args.direct_answer_contrast_weight,
        params=ctx.params(),
        direct_step=direct_step,
        direct_baseline=ctx.direct_baseline,
        snapshot_recorder=ctx.snapshot_recorder,
        update_guard=ctx.update_guard,
        update_shape=update_shape,
        profile=profile,
        profile_frontier_records=frontier_records,
        restore_direct_update_state=ctx.restore_direct_update_state,
        diversity_outcome=state.diversity_outcome,
        diversity_rejection_reason=state.diversity_rejection_reason,
    )
    state.apply_branch_diversity_recovery(result)
    return result


def try_profile_scale_collapsed_profile_binding(
    ctx: Any,
    state: BaselineFloorProfileAttemptState,
    profile: str,
    profile_batch: list[Any],
    frontier_records: int,
    diversity_ok: bool,
    coverage_ok: bool,
    profile_scale: float,
    update_shape: str,
    direct_step: int,
) -> Any:
    setup = ctx.direct_setup
    result = try_baseline_floor_collapsed_profile_binding(
        active=setup.direct_answer_baseline_floor_profile_scale_collapsed_profile_binding_frontier_stabilization_active,
        floor_preserved=state.floor_preserved,
        diversity_accepted=diversity_ok,
        coverage_accepted=coverage_ok,
        profile_score=state.profile_score,
        profile_probe_snapshot=state.profile_probe_snapshot,
        model=ctx.model(),
        tokenizer=ctx.tokenizer(),
        optimizer=ctx.optimizer(),
        profile_batch=profile_batch,
        base_learning_rate=ctx.args.direct_answer_learning_rate,
        profile_scale=profile_scale,
        negative_weight=ctx.args.direct_answer_negative_weight,
        positive_weight=ctx.args.direct_answer_positive_weight,
        contrast_weight=ctx.args.direct_answer_contrast_weight,
        params=ctx.params(),
        direct_step=direct_step,
        direct_baseline=ctx.direct_baseline,
        snapshot_recorder=ctx.snapshot_recorder,
        update_guard=ctx.update_guard,
        update_shape=update_shape,
        profile=profile,
        profile_frontier_records=frontier_records,
        restore_direct_update_state=ctx.restore_direct_update_state,
        diversity_outcome=state.diversity_outcome,
        diversity_rejection_reason=state.diversity_rejection_reason,
        owner_paraphrase_binding_active=setup.direct_answer_baseline_floor_profile_scale_owner_paraphrase_binding_frontier_stabilization_active,
        owner_paraphrase_target_profiles=(
            BASELINE_FLOOR_OWNER_PARAPHRASE_BINDING_TARGET_PROFILES
        ),
        owner_paraphrase_preserved_profiles=(
            BASELINE_FLOOR_OWNER_PARAPHRASE_BINDING_PRESERVED_PROFILES
        ),
        owner_paraphrase_binding_preservation_delta=(
            state.owner_paraphrase_binding_preservation_delta
        ),
        memory_consolidation_active=setup.direct_answer_baseline_floor_profile_scale_memory_consolidation_frontier_stabilization_active,
        memory_consolidation_target_profiles=(
            setup.direct_memory_consolidation_target_profiles
        ),
    )
    state.apply_collapsed_profile_binding(result)
    return result


def try_profile_scale_missing_first_token(
    ctx: Any,
    state: BaselineFloorProfileAttemptState,
    profile: str,
    profile_batch: list[Any],
    frontier_records: int,
    priorities: dict[str, bool],
    profile_base_snapshot: dict[str, Any] | None,
    diversity_ok: bool,
    coverage_ok: bool,
    profile_scale: float,
    update_shape: str,
    direct_step: int,
) -> Any:
    setup = ctx.direct_setup
    result = try_baseline_floor_missing_first_token_consolidation(
        active=setup.direct_answer_baseline_floor_profile_scale_missing_first_token_consolidation_frontier_stabilization_active,
        memory_consolidation_prioritized=priorities["memory"],
        floor_preserved=state.floor_preserved,
        diversity_accepted=diversity_ok,
        coverage_accepted=coverage_ok,
        profile_score=state.profile_score,
        profile_probe_snapshot=state.profile_probe_snapshot,
        coverage_delta=state.coverage_delta,
        coverage_outcome=state.coverage_outcome,
        coverage_rejection_reason=state.coverage_rejection_reason,
        profile_base_snapshot=profile_base_snapshot,
        model=ctx.model(),
        tokenizer=ctx.tokenizer(),
        optimizer=ctx.optimizer(),
        profile_batch=profile_batch,
        rng=ctx.rng,
        base_learning_rate=ctx.args.direct_answer_learning_rate,
        profile_scale=profile_scale,
        negative_weight=ctx.args.direct_answer_negative_weight,
        positive_weight=ctx.args.direct_answer_positive_weight,
        contrast_weight=ctx.args.direct_answer_contrast_weight,
        params=ctx.params(),
        direct_step=direct_step,
        direct_baseline=ctx.direct_baseline,
        snapshot_recorder=ctx.snapshot_recorder,
        update_guard=ctx.update_guard,
        update_shape=update_shape,
        profile=profile,
        profile_frontier_records=frontier_records,
        target_profiles=setup.direct_memory_consolidation_target_profiles,
        missing_first_token_ids_by_profile=(
            setup.direct_memory_consolidation_missing_first_token_ids
        ),
        profile_specific=setup.direct_answer_baseline_floor_profile_scale_remaining_collapsed_profile_specific_missing_first_token_consolidation_frontier_stabilization_active,
        restore_direct_update_state=ctx.restore_direct_update_state,
        diversity_outcome=state.diversity_outcome,
        diversity_rejection_reason=state.diversity_rejection_reason,
    )
    state.apply_missing_first_token(result)
    return result

