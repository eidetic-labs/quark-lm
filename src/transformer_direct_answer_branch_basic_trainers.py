"""Objective adapters for basic branch direct-answer modes."""

from __future__ import annotations

from transformer_direct_answer_branch_basic_objectives import (
    train_direct_answer_branch_batch_contrast_unlikelihood,
    train_direct_answer_branch_collapse_unlikelihood,
    train_direct_answer_branch_diversity_unlikelihood,
    train_direct_answer_branch_hidden_projection_margin_unlikelihood,
    train_direct_answer_branch_target_margin_unlikelihood,
    train_direct_answer_branch_target_softmax_unlikelihood,
)
from transformer_direct_answer_branch_basic_step import BranchBasicModeStep
from transformer_direct_answer_repair_objectives import (
    train_direct_answer_branch_repair_unlikelihood,
)
from transformer_direct_answer_repairs import train_direct_answer_first_error_unlikelihood


def train_first_error_unlikelihood(step: BranchBasicModeStep) -> float:
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


def train_branch_repair(step: BranchBasicModeStep) -> float:
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


def train_branch_collapse(step: BranchBasicModeStep) -> float:
    args = step.args
    return train_direct_answer_branch_collapse_unlikelihood(
        step.model,
        step.tokenizer,
        step.example,
        step.branch_examples,
        step.lesson,
        step.rng,
        args.direct_answer_learning_rate,
        args.direct_answer_negative_weight,
        args.direct_answer_positive_weight,
        args.direct_answer_branch_position,
        args.direct_answer_hard_negatives,
        step.terminator,
        step.params,
    )


def train_branch_batch_contrast(step: BranchBasicModeStep) -> float:
    args = step.args
    return train_direct_answer_branch_batch_contrast_unlikelihood(
        step.model,
        step.tokenizer,
        step.example,
        step.branch_examples,
        step.lesson,
        step.rng,
        args.direct_answer_learning_rate,
        args.direct_answer_negative_weight,
        args.direct_answer_positive_weight,
        args.direct_answer_branch_position,
        args.direct_answer_branch_batch_size,
        step.terminator,
        step.params,
    )


def train_branch_diversity(step: BranchBasicModeStep) -> float:
    args = step.args
    return train_direct_answer_branch_diversity_unlikelihood(
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


def train_branch_target_softmax(step: BranchBasicModeStep) -> float:
    args = step.args
    return train_direct_answer_branch_target_softmax_unlikelihood(
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


def train_branch_target_margin(step: BranchBasicModeStep) -> float:
    args = step.args
    return train_direct_answer_branch_target_margin_unlikelihood(
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


def train_branch_hidden_projection_margin(step: BranchBasicModeStep) -> float:
    args = step.args
    return train_direct_answer_branch_hidden_projection_margin_unlikelihood(
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
