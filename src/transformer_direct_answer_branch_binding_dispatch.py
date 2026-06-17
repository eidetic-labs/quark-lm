"""Dispatch branch binding and target-replay modes for direct-answer training."""

from __future__ import annotations

import argparse
import random
from typing import Any

from answer_model import AnswerExample
from autograd import Scalar
from tokenizer import CharTokenizer
from transformer_direct_answer_branch_binding_objectives import (
    train_direct_answer_branch_bidirectional_binding_unlikelihood,
    train_direct_answer_branch_coverage_binding_unlikelihood,
    train_direct_answer_branch_output_binding_unlikelihood,
    train_direct_answer_branch_representation_contrast_unlikelihood,
    train_direct_answer_branch_target_diversity_unlikelihood,
    train_direct_answer_branch_target_replay_coverage_unlikelihood,
    train_direct_answer_branch_target_set_coverage_unlikelihood,
)
from transformer_direct_answer_core import DirectAnswerLesson


BRANCH_BINDING_DIRECT_ANSWER_MODES = frozenset(
    {
        "branch-representation-contrast-unlikelihood",
        "branch-balanced-representation-contrast-unlikelihood",
        "branch-output-binding-unlikelihood",
        "branch-bidirectional-binding-unlikelihood",
        "branch-balanced-bidirectional-binding-unlikelihood",
        "branch-coverage-binding-unlikelihood",
        "branch-balanced-coverage-binding-unlikelihood",
        "branch-target-set-coverage-unlikelihood",
        "branch-balanced-target-set-coverage-unlikelihood",
        "branch-target-diversity-unlikelihood",
        "branch-balanced-target-diversity-unlikelihood",
        "branch-target-replay-coverage-unlikelihood",
        "branch-balanced-target-replay-coverage-unlikelihood",
    }
)


def train_direct_answer_branch_binding_mode_step(
    *,
    args: argparse.Namespace,
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    lesson: DirectAnswerLesson,
    branch_examples: list[AnswerExample],
    rng: random.Random,
    terminator: str,
    params: list[Scalar],
) -> float:
    mode = args.direct_answer_mode
    base_args = _base_branch_args(args, model, tokenizer, example, branch_examples, lesson, rng)
    balance_targets = _uses_balanced_targets(mode)

    if mode in _REPRESENTATION_CONTRAST_MODES:
        return train_direct_answer_branch_representation_contrast_unlikelihood(
            *base_args,
            terminator,
            params,
            balance_targets=balance_targets,
        )
    if mode == "branch-output-binding-unlikelihood":
        return train_direct_answer_branch_output_binding_unlikelihood(
            *base_args,
            terminator,
            params,
        )
    if mode in _BIDIRECTIONAL_BINDING_MODES:
        return train_direct_answer_branch_bidirectional_binding_unlikelihood(
            *base_args,
            terminator,
            params,
            balance_targets=balance_targets,
        )
    if mode in _COVERAGE_BINDING_MODES:
        return train_direct_answer_branch_coverage_binding_unlikelihood(
            *base_args,
            args.direct_answer_hard_negatives,
            terminator,
            params,
            balance_targets=balance_targets,
        )
    if mode in _TARGET_SET_COVERAGE_MODES:
        return train_direct_answer_branch_target_set_coverage_unlikelihood(
            *base_args,
            args.direct_answer_hard_negatives,
            terminator,
            params,
            balance_targets=balance_targets,
        )
    if mode in _TARGET_DIVERSITY_MODES:
        return train_direct_answer_branch_target_diversity_unlikelihood(
            *base_args,
            args.direct_answer_hard_negatives,
            terminator,
            params,
            balance_targets=balance_targets,
        )
    if mode in _TARGET_REPLAY_COVERAGE_MODES:
        return train_direct_answer_branch_target_replay_coverage_unlikelihood(
            *base_args,
            args.direct_answer_hard_negatives,
            terminator,
            params,
            balance_targets=balance_targets,
        )
    raise ValueError(f"Unsupported branch binding direct-answer mode: {mode}")


def _base_branch_args(
    args: argparse.Namespace,
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    lesson: DirectAnswerLesson,
    rng: random.Random,
) -> tuple[Any, ...]:
    return (
        model,
        tokenizer,
        example,
        branch_examples,
        lesson,
        rng,
        args.direct_answer_learning_rate,
        args.direct_answer_negative_weight,
        args.direct_answer_positive_weight,
        args.direct_answer_contrast_weight,
        args.direct_answer_branch_position,
        args.direct_answer_branch_batch_size,
    )


def _uses_balanced_targets(mode: str) -> bool:
    return mode.startswith("branch-balanced-")


_REPRESENTATION_CONTRAST_MODES = frozenset(
    {
        "branch-representation-contrast-unlikelihood",
        "branch-balanced-representation-contrast-unlikelihood",
    }
)
_BIDIRECTIONAL_BINDING_MODES = frozenset(
    {
        "branch-bidirectional-binding-unlikelihood",
        "branch-balanced-bidirectional-binding-unlikelihood",
    }
)
_COVERAGE_BINDING_MODES = frozenset(
    {
        "branch-coverage-binding-unlikelihood",
        "branch-balanced-coverage-binding-unlikelihood",
    }
)
_TARGET_SET_COVERAGE_MODES = frozenset(
    {
        "branch-target-set-coverage-unlikelihood",
        "branch-balanced-target-set-coverage-unlikelihood",
    }
)
_TARGET_DIVERSITY_MODES = frozenset(
    {
        "branch-target-diversity-unlikelihood",
        "branch-balanced-target-diversity-unlikelihood",
    }
)
_TARGET_REPLAY_COVERAGE_MODES = frozenset(
    {
        "branch-target-replay-coverage-unlikelihood",
        "branch-balanced-target-replay-coverage-unlikelihood",
    }
)
