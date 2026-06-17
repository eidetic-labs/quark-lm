"""Context-replay coverage objective update helper."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from answer_model import AnswerExample
from autograd import Scalar
from replay_plan import BranchReplayRecord
from tokenizer import CharTokenizer
from transformer_baseline_floor_anchor_selection import (
    baseline_floor_objective_anchor_batch,
)
from transformer_direct_answer_batches import (
    direct_answer_branch_diversity_batch,
    direct_answer_profiled_branch_batch,
    direct_answer_target_balanced_branch_diversity_batch,
)
from transformer_direct_answer_core import DirectAnswerLesson
from transformer_direct_answer_repairs import train_direct_answer_lesson
from transformer_direct_modes import (
    ANSWER_TERMINATOR,
    BASELINE_FLOOR_OBJECTIVE_ANCHOR_BATCH_SIZE,
    BASELINE_FLOOR_OBJECTIVE_ANCHOR_WEIGHT,
    ReplayPredictionOverrides,
)


@dataclass
class DirectAnswerBaselineAnchoredPromptUpdater:
    model: Callable[[], Any]
    tokenizer: Callable[[], CharTokenizer]
    training_pool: list[AnswerExample]
    rng: random.Random
    update_guard: dict[str, Any]
    negative_weight: float
    positive_weight: float
    contrast_weight: float
    branch_position: int
    branch_batch_size: int
    hard_negatives: int
    terminator: str
    params: Callable[[], list[Scalar]]
    replay_prediction_overrides: ReplayPredictionOverrides | None
    baseline_floor_objective_active: bool
    baseline_floor_repair_anchors: list[BranchReplayRecord]

    def train(
        self,
        example: AnswerExample,
        fallback_lesson: DirectAnswerLesson,
        learning_rate: float,
    ) -> float:
        floor_preservation_branches: list[BranchReplayRecord] | None = None
        if self.baseline_floor_objective_active and self.baseline_floor_repair_anchors:
            floor_preservation_branches = baseline_floor_objective_anchor_batch(
                self.baseline_floor_repair_anchors,
                self.rng,
                BASELINE_FLOOR_OBJECTIVE_ANCHOR_BATCH_SIZE,
            )
            self.update_guard["objective_anchor_batches"] += 1
            self.update_guard["objective_anchor_records"] += len(
                floor_preservation_branches
            )
        return train_direct_answer_branch_context_replay_coverage_unlikelihood(
            self.model(),
            self.tokenizer(),
            example,
            self.training_pool,
            fallback_lesson,
            self.rng,
            learning_rate,
            self.negative_weight,
            self.positive_weight,
            self.contrast_weight,
            self.branch_position,
            self.branch_batch_size,
            self.hard_negatives,
            self.terminator,
            self.params(),
            balance_targets=True,
            focus_uncovered_targets=True,
            preserve_predicted_target_coverage=True,
            balance_deficit_targets=True,
            profile_aware_targets=True,
            balance_profile_target_shares=True,
            enforce_prompt_target_margins=True,
            replay_prediction_overrides=self.replay_prediction_overrides,
            floor_preservation_branches=floor_preservation_branches,
            floor_preservation_weight=(
                BASELINE_FLOOR_OBJECTIVE_ANCHOR_WEIGHT
                if self.baseline_floor_objective_active
                else 0.0
            ),
            balance_floor_preservation_targets=self.baseline_floor_objective_active,
        )


def train_direct_answer_branch_context_replay_coverage_unlikelihood(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    replay_weight: float,
    branch_position: int,
    batch_size: int,
    hard_negative_count: int,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
    balance_targets: bool = False,
    preserve_covered_targets: bool = False,
    balance_covered_target_anchors: bool = False,
    focus_uncovered_targets: bool = False,
    preserve_predicted_target_coverage: bool = False,
    balance_deficit_targets: bool = False,
    profile_aware_targets: bool = False,
    balance_profile_target_shares: bool = False,
    enforce_prompt_target_margins: bool = False,
    replay_prediction_overrides: ReplayPredictionOverrides | None = None,
    floor_preservation_branches: list[BranchReplayRecord] | None = None,
    floor_preservation_weight: float = 0.0,
    balance_floor_preservation_targets: bool = False,
) -> float:
    if profile_aware_targets:
        branches = direct_answer_profiled_branch_batch(
            model,
            tokenizer,
            example,
            branch_examples,
            rng,
            branch_position,
            batch_size,
            terminator,
            balance_targets=balance_targets,
        )
    else:
        batch_builder = (
            direct_answer_target_balanced_branch_diversity_batch
            if balance_targets
            else direct_answer_branch_diversity_batch
        )
        branches = batch_builder(
            model,
            tokenizer,
            example,
            branch_examples,
            rng,
            branch_position,
            batch_size,
            terminator,
        )
    if not branches:
        return train_direct_answer_lesson(
            model,
            fallback_lesson,
            rng,
            learning_rate,
            params=params,
        )
    replay_batch_size = max(batch_size, batch_size + max(0, hard_negative_count))
    if profile_aware_targets:
        replay_branches = direct_answer_profiled_branch_batch(
            model,
            tokenizer,
            example,
            branch_examples,
            rng,
            branch_position,
            replay_batch_size,
            terminator,
            balance_targets=True,
            prediction_overrides=replay_prediction_overrides,
        )
    else:
        replay_branches = direct_answer_target_balanced_branch_diversity_batch(
            model,
            tokenizer,
            example,
            branch_examples,
            rng,
            branch_position,
            replay_batch_size,
            terminator,
        )
    return model.train_step_with_branch_context_replay_coverage(
        branches,
        replay_branches,
        learning_rate,
        negative_weight,
        positive_weight,
        replay_weight,
        hard_negative_count,
        params=params,
        preserve_covered_targets=preserve_covered_targets,
        balance_covered_target_anchors=balance_covered_target_anchors,
        focus_uncovered_targets=focus_uncovered_targets,
        preserve_predicted_target_coverage=preserve_predicted_target_coverage,
        balance_deficit_targets=balance_deficit_targets,
        profile_aware_targets=profile_aware_targets,
        balance_profile_target_shares=balance_profile_target_shares,
        enforce_prompt_target_margins=enforce_prompt_target_margins,
        floor_preservation_branches=floor_preservation_branches,
        floor_preservation_weight=floor_preservation_weight,
        balance_floor_preservation_targets=balance_floor_preservation_targets,
    )
