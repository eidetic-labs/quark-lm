"""Baseline-floor profile-scale stabilization stage."""

from __future__ import annotations

from typing import Any

from transformer_baseline_floor_anchor_selection import baseline_floor_profile_setup
from transformer_baseline_floor_profile_scale_attempts import (
    train_baseline_floor_profile_attempts,
)
from transformer_baseline_floor_profile_scale_planning import (
    record_baseline_floor_profile_scale_remaining_sources,
)
from transformer_baseline_floor_update_shapes import baseline_floor_attempt_update_shape


def train_baseline_floor_profile_scale_stage(
    ctx: Any,
    direct_step: int,
) -> tuple[float, bool]:
    setup = ctx.direct_setup
    profile_setup = baseline_floor_profile_setup(
        repair_anchors=setup.direct_baseline_floor_repair_anchors,
        frontier_anchors=setup.direct_baseline_floor_frontier_anchors,
        remaining_binding_target_profiles=(
            setup.direct_remaining_profile_binding_target_profiles
        ),
        include_frontier_anchors=(
            setup.direct_answer_baseline_floor_profile_scale_frontier_stabilization_active
        ),
        prioritize_remaining_profile_binding=(
            setup.direct_answer_baseline_floor_profile_scale_remaining_profile_binding_frontier_stabilization_active
        ),
    )
    remaining_sources = profile_setup.remaining_source_profiles
    record_baseline_floor_profile_scale_remaining_sources(ctx, remaining_sources)
    update_shape = baseline_floor_attempt_update_shape(ctx.args.direct_answer_mode)
    total_loss = 0.0
    loss_count = 0
    accepted_any = False
    for profile, profile_anchors in profile_setup.profile_items:
        loss, count, accepted = train_baseline_floor_profile_attempts(
            ctx,
            profile=profile,
            profile_anchors=profile_anchors,
            remaining_source_profiles=remaining_sources,
            frontier_targets_by_profile=profile_setup.frontier_targets_by_profile,
            update_shape=update_shape,
            direct_step=direct_step,
        )
        total_loss += loss
        loss_count += count
        accepted_any = accepted_any or accepted
    return total_loss / max(loss_count, 1), accepted_any
