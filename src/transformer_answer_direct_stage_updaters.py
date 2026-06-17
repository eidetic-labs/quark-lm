"""Updater construction for direct-answer stage orchestration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from transformer_baseline_floor_stabilization_stage import (
    BaselineFloorStabilizationContext,
    train_baseline_floor_stabilization_stage,
)
from transformer_direct_answer_context_replay_objective import (
    DirectAnswerBaselineAnchoredPromptUpdater,
)
from transformer_answer_direct_stage_state import DirectAnswerStageState


def build_baseline_anchored_prompt_updater(
    *,
    args: Any,
    direct_setup: Any,
    direct_runtime: Any,
    stage_state: DirectAnswerStageState,
) -> DirectAnswerBaselineAnchoredPromptUpdater:
    return DirectAnswerBaselineAnchoredPromptUpdater(
        model=lambda: stage_state.model,
        tokenizer=lambda: stage_state.tokenizer,
        training_pool=direct_setup.direct_training_pool,
        rng=direct_setup.direct_rng,
        update_guard=direct_runtime.update_guard,
        negative_weight=args.direct_answer_negative_weight,
        positive_weight=args.direct_answer_positive_weight,
        contrast_weight=args.direct_answer_contrast_weight,
        branch_position=args.direct_answer_branch_position,
        branch_batch_size=args.direct_answer_branch_batch_size,
        hard_negatives=args.direct_answer_hard_negatives,
        terminator=direct_setup.direct_answer_terminator,
        params=lambda: stage_state.params,
        replay_prediction_overrides=(
            direct_setup.direct_replay_prediction_overrides
        ),
        baseline_floor_objective_active=(
            direct_setup.direct_answer_baseline_floor_objective_active
        ),
        baseline_floor_repair_anchors=direct_setup.direct_baseline_floor_repair_anchors,
    )


def build_stabilization_context(
    *,
    args: Any,
    direct_setup: Any,
    direct_runtime: Any,
    stage_state: DirectAnswerStageState,
) -> BaselineFloorStabilizationContext:
    return BaselineFloorStabilizationContext(
        args=args,
        direct_setup=direct_setup,
        model=lambda: stage_state.model,
        tokenizer=lambda: stage_state.tokenizer,
        optimizer=lambda: stage_state.optimizer,
        rng=direct_setup.direct_rng,
        params=lambda: stage_state.params,
        update_guard=direct_runtime.update_guard,
        direct_baseline=direct_runtime.baseline,
        snapshot_recorder=direct_runtime.snapshot_recorder,
        restore_direct_update_state=stage_state.restore,
    )


def build_stabilization_trainer(
    context: BaselineFloorStabilizationContext,
) -> Callable[[float, int], tuple[float, bool]]:
    def train_baseline_floor_stabilization_update(
        update_learning_rate: float,
        direct_step: int,
    ) -> tuple[float, bool]:
        return train_baseline_floor_stabilization_stage(
            context,
            update_learning_rate,
            direct_step,
        )

    return train_baseline_floor_stabilization_update
