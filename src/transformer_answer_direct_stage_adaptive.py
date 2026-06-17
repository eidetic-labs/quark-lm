"""Adaptive baseline-floor update construction for direct-answer stages."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from transformer_baseline_floor_adaptive import (
    train_adaptive_baseline_floor_update_stage,
)
from transformer_answer_direct_stage_state import DirectAnswerStageState


def build_adaptive_baseline_floor_trainer(
    *,
    args: Any,
    direct_setup: Any,
    direct_runtime: Any,
    stage_state: DirectAnswerStageState,
    train_stabilization_update: Callable[[float, int], tuple[float, bool]],
    train_baseline_anchored_prompt: Callable[..., float],
) -> Callable[..., float]:
    def train_adaptive_baseline_floor_update(
        example: Any,
        direct_step: int,
        base_model_payload: dict[str, Any],
        base_optimizer_payload: dict[str, Any],
        base_rng_state: object,
    ) -> float:
        return train_adaptive_baseline_floor_update_stage(
            model=stage_state.model,
            rng=direct_setup.direct_rng,
            example=example,
            lesson=direct_setup.direct_lessons[example],
            direct_step=direct_step,
            base_model_payload=base_model_payload,
            base_optimizer_payload=base_optimizer_payload,
            base_rng_state=base_rng_state,
            direct_answer_mode=args.direct_answer_mode,
            direct_answer_learning_rate=args.direct_answer_learning_rate,
            direct_answer_branch_batch_size=args.direct_answer_branch_batch_size,
            direct_answer_hard_negatives=args.direct_answer_hard_negatives,
            update_guard=direct_runtime.update_guard,
            direct_baseline=direct_runtime.baseline,
            snapshot_recorder=direct_runtime.snapshot_recorder,
            outer_learning_rate_scales=(
                direct_setup.direct_baseline_floor_outer_learning_rate_scales
            ),
            repair_anchors=direct_setup.direct_baseline_floor_repair_anchors,
            repaired_updates_active=(
                direct_setup.direct_answer_baseline_floor_repaired_updates_active
            ),
            stabilization_active=(
                direct_setup.direct_answer_baseline_floor_stabilization_active
            ),
            profile_scale_diversity_active=(
                direct_setup.direct_answer_baseline_floor_profile_scale_diversity_stabilization_active
            ),
            train_stabilization_update=train_stabilization_update,
            train_baseline_anchored_prompt=train_baseline_anchored_prompt,
            restore_direct_update_state=stage_state.restore,
            params=stage_state.params,
        )

    return train_adaptive_baseline_floor_update
