"""Objective adapters for branch-contrast direct-answer modes."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Any

from answer_model import AnswerExample
from autograd import Scalar
from tokenizer import CharTokenizer
from transformer_direct_answer_branch_binding_objectives import (
    train_direct_answer_branch_representation_contrast_unlikelihood,
)
from transformer_direct_answer_branch_contrast_hard_objective import (
    train_direct_answer_hard_branch_contrast_unlikelihood,
)
from transformer_direct_answer_branch_contrast_pair_objective import (
    train_direct_answer_branch_contrast_unlikelihood,
)
from transformer_direct_answer_branch_contrast_span_objective import (
    train_direct_answer_branch_span_contrast_unlikelihood,
)
from transformer_direct_answer_branch_ranking_objectives import (
    train_direct_answer_branch_rank_margin_unlikelihood,
    train_direct_answer_branch_topk_softmax_unlikelihood,
)
from transformer_direct_answer_core import DirectAnswerLesson
from transformer_direct_answer_repair_objectives import (
    train_direct_answer_branch_repair_unlikelihood,
    train_direct_answer_branch_span_repair_unlikelihood,
)
from transformer_direct_answer_repairs import train_direct_answer_first_error_unlikelihood


@dataclass(frozen=True)
class BranchContrastModeStep:
    args: argparse.Namespace
    model: Any
    tokenizer: CharTokenizer
    example: AnswerExample
    lesson: DirectAnswerLesson
    branch_examples: list[AnswerExample]
    rng: random.Random
    terminator: str
    params: list[Scalar]


def train_first_error_unlikelihood(step: BranchContrastModeStep) -> float:
    args = step.args
    return train_direct_answer_first_error_unlikelihood(
        step.model,
        step.tokenizer,
        step.example,
        step.lesson,
        step.rng,
        args.direct_answer_learning_rate,
        args.direct_answer_negative_weight,
        step.terminator,
        step.params,
    )


def train_branch_repair(step: BranchContrastModeStep) -> float:
    args = step.args
    return train_direct_answer_branch_repair_unlikelihood(
        step.model,
        step.tokenizer,
        step.example,
        step.lesson,
        step.rng,
        args.direct_answer_learning_rate,
        args.direct_answer_negative_weight,
        args.direct_answer_positive_weight,
        args.direct_answer_branch_position,
        step.terminator,
        step.params,
    )


def train_branch_span_repair(step: BranchContrastModeStep) -> float:
    args = step.args
    return train_direct_answer_branch_span_repair_unlikelihood(
        step.model,
        step.tokenizer,
        step.example,
        step.lesson,
        step.rng,
        args.direct_answer_learning_rate,
        args.direct_answer_negative_weight,
        args.direct_answer_positive_weight,
        args.direct_answer_branch_position,
        args.direct_answer_branch_span,
        step.terminator,
        step.params,
    )


def train_branch_rank_margin(
    step: BranchContrastModeStep,
    *,
    balance_targets: bool,
) -> float:
    args = step.args
    return train_direct_answer_branch_rank_margin_unlikelihood(
        step.model,
        step.tokenizer,
        step.example,
        step.branch_examples,
        step.lesson,
        step.rng,
        args.direct_answer_learning_rate,
        args.direct_answer_negative_weight,
        args.direct_answer_positive_weight,
        args.direct_answer_contrast_weight,
        args.direct_answer_branch_position,
        args.direct_answer_branch_batch_size,
        args.direct_answer_hard_negatives,
        step.terminator,
        step.params,
        balance_targets=balance_targets,
    )


def train_branch_topk_softmax(
    step: BranchContrastModeStep,
    *,
    balance_targets: bool,
) -> float:
    args = step.args
    return train_direct_answer_branch_topk_softmax_unlikelihood(
        step.model,
        step.tokenizer,
        step.example,
        step.branch_examples,
        step.lesson,
        step.rng,
        args.direct_answer_learning_rate,
        args.direct_answer_negative_weight,
        args.direct_answer_positive_weight,
        args.direct_answer_contrast_weight,
        args.direct_answer_branch_position,
        args.direct_answer_branch_batch_size,
        args.direct_answer_hard_negatives,
        step.terminator,
        step.params,
        balance_targets=balance_targets,
    )


def train_branch_representation_contrast(step: BranchContrastModeStep) -> float:
    args = step.args
    return train_direct_answer_branch_representation_contrast_unlikelihood(
        step.model,
        step.tokenizer,
        step.example,
        step.branch_examples,
        step.lesson,
        step.rng,
        args.direct_answer_learning_rate,
        args.direct_answer_negative_weight,
        args.direct_answer_positive_weight,
        args.direct_answer_contrast_weight,
        args.direct_answer_branch_position,
        args.direct_answer_branch_batch_size,
        step.terminator,
        step.params,
    )


def train_branch_contrast(step: BranchContrastModeStep) -> float:
    args = step.args
    return train_direct_answer_branch_contrast_unlikelihood(
        step.model,
        step.tokenizer,
        step.example,
        step.branch_examples,
        step.lesson,
        step.rng,
        args.direct_answer_learning_rate,
        args.direct_answer_negative_weight,
        args.direct_answer_contrast_weight,
        args.direct_answer_branch_position,
        step.terminator,
        step.params,
    )


def train_branch_span_contrast(step: BranchContrastModeStep) -> float:
    args = step.args
    return train_direct_answer_branch_span_contrast_unlikelihood(
        step.model,
        step.tokenizer,
        step.example,
        step.branch_examples,
        step.lesson,
        step.rng,
        args.direct_answer_learning_rate,
        args.direct_answer_negative_weight,
        args.direct_answer_positive_weight,
        args.direct_answer_contrast_weight,
        args.direct_answer_branch_position,
        args.direct_answer_branch_span,
        step.terminator,
        step.params,
    )


def train_hard_branch_contrast(step: BranchContrastModeStep) -> float:
    args = step.args
    return train_direct_answer_hard_branch_contrast_unlikelihood(
        step.model,
        step.tokenizer,
        step.example,
        step.branch_examples,
        step.lesson,
        step.rng,
        args.direct_answer_learning_rate,
        args.direct_answer_negative_weight,
        args.direct_answer_positive_weight,
        args.direct_answer_contrast_weight,
        args.direct_answer_branch_position,
        args.direct_answer_hard_negatives,
        step.terminator,
        step.params,
    )
