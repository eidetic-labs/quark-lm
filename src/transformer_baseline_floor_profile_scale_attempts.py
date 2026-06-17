"""Profile-scale baseline-floor attempt execution."""

from __future__ import annotations

from typing import Any

from transformer_baseline_floor_training import (
    train_direct_answer_baseline_floor_anchor_batch,
)
from transformer_baseline_floor_profile_scale_accounting import (
    record_profile_scale_acceptance,
    record_profile_scale_rejection,
)
from transformer_baseline_floor_profile_scale_planning import (
    baseline_floor_profile_scale_priorities,
    select_baseline_floor_profile_scale_batch,
)
from transformer_baseline_floor_profile_scale_probes import (
    initial_baseline_floor_profile_attempt_state,
    record_baseline_floor_profile_base_probe,
    record_baseline_floor_profile_scale_probe,
)
from transformer_baseline_floor_profile_scale_recovery import (
    apply_profile_scale_recovery_attempts,
)


def train_baseline_floor_profile_attempts(
    ctx: Any,
    *,
    profile: str,
    profile_anchors: list[Any],
    remaining_source_profiles: list[str],
    frontier_targets_by_profile: dict[str, set[int]],
    update_shape: str,
    direct_step: int,
) -> tuple[float, int, bool]:
    setup = ctx.direct_setup
    priorities = baseline_floor_profile_scale_priorities(
        setup,
        profile,
        remaining_source_profiles,
    )
    model_payload = ctx.model().to_dict(ctx.tokenizer())
    optimizer_payload = ctx.optimizer().to_dict()
    rng_state = ctx.rng.getstate()
    profile_base_snapshot, profile_base_score = record_baseline_floor_profile_base_probe(
        ctx,
        profile,
        update_shape,
        direct_step,
    )
    total_loss = 0.0
    loss_count = 0
    for profile_scale in setup.direct_baseline_floor_learning_rate_scales:
        ctx.restore_direct_update_state(model_payload, optimizer_payload)
        ctx.rng.setstate(rng_state)
        loss, count, accepted = _train_profile_scale_attempt(
            ctx,
            profile=profile,
            profile_anchors=profile_anchors,
            remaining_source_profiles=remaining_source_profiles,
            frontier_targets_by_profile=frontier_targets_by_profile,
            priorities=priorities,
            profile_base_snapshot=profile_base_snapshot,
            profile_base_score=profile_base_score,
            profile_scale=profile_scale,
            update_shape=update_shape,
            direct_step=direct_step,
        )
        total_loss += loss
        loss_count += count
        if accepted:
            return total_loss, loss_count, True
    ctx.restore_direct_update_state(model_payload, optimizer_payload)
    ctx.rng.setstate(rng_state)
    return total_loss, loss_count, False


def _train_profile_scale_attempt(
    ctx: Any,
    *,
    profile: str,
    profile_anchors: list[Any],
    remaining_source_profiles: list[str],
    frontier_targets_by_profile: dict[str, set[int]],
    priorities: dict[str, bool],
    profile_base_snapshot: dict[str, Any] | None,
    profile_base_score: tuple[float, ...] | None,
    profile_scale: float,
    update_shape: str,
    direct_step: int,
) -> tuple[float, int, bool]:
    profile_batch, frontier_records = select_baseline_floor_profile_scale_batch(
        ctx,
        profile=profile,
        profile_anchors=profile_anchors,
        frontier_targets_by_profile=frontier_targets_by_profile,
        priorities=priorities,
    )
    loss = train_direct_answer_baseline_floor_anchor_batch(
        ctx.model(),
        profile_batch,
        ctx.args.direct_answer_learning_rate * profile_scale,
        params=ctx.params(),
    )
    snapshot = record_baseline_floor_profile_scale_probe(
        ctx,
        profile=profile,
        profile_batch=profile_batch,
        frontier_records=frontier_records,
        profile_scale=profile_scale,
        update_shape=update_shape,
        priorities=priorities,
        direct_step=direct_step,
    )
    state = initial_baseline_floor_profile_attempt_state(
        ctx,
        snapshot=snapshot,
        profile_base_snapshot=profile_base_snapshot,
        profile_base_score=profile_base_score,
    )
    extra_loss, extra_count, diversity_ok, coverage_ok = (
        apply_profile_scale_recovery_attempts(
            ctx,
            state=state,
            profile=profile,
            profile_batch=profile_batch,
            frontier_targets_by_profile=frontier_targets_by_profile,
            frontier_records=frontier_records,
            priorities=priorities,
            profile_base_snapshot=profile_base_snapshot,
            profile_base_score=profile_base_score,
            profile_scale=profile_scale,
            update_shape=update_shape,
            direct_step=direct_step,
        )
    )
    total_loss = loss + extra_loss
    total_count = 1 + extra_count
    if state.accepted(diversity_ok, coverage_ok):
        record_profile_scale_acceptance(
            ctx,
            state=state,
            profile=profile,
            records=len(profile_batch),
            frontier_records=frontier_records,
            profile_scale=profile_scale,
            priorities=priorities,
            remaining_source_profiles=remaining_source_profiles,
        )
        return total_loss, total_count, True
    record_profile_scale_rejection(
        ctx,
        state=state,
        profile=profile,
        records=len(profile_batch),
        frontier_records=frontier_records,
        profile_scale=profile_scale,
        priorities=priorities,
        diversity_ok=diversity_ok,
        profile_base_score=profile_base_score,
    )
    return total_loss, total_count, False
