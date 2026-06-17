"""Shared batch and fallback helpers for branch-binding objectives."""

from __future__ import annotations

import random
from typing import Any

from answer_model import AnswerExample
from autograd import Scalar
from tokenizer import CharTokenizer
from transformer_direct_answer_batches import (
    direct_answer_branch_diversity_batch,
    direct_answer_target_balanced_branch_diversity_batch,
)
from transformer_direct_answer_core import DirectAnswerLesson
from transformer_direct_answer_repairs import train_direct_answer_lesson
from transformer_direct_modes import ANSWER_TERMINATOR


def branch_binding_batch(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    rng: random.Random,
    branch_position: int,
    batch_size: int,
    terminator: str = ANSWER_TERMINATOR,
    balance_targets: bool = False,
) -> list[Any]:
    batch_builder = (
        direct_answer_target_balanced_branch_diversity_batch
        if balance_targets
        else direct_answer_branch_diversity_batch
    )
    return batch_builder(
        model,
        tokenizer,
        example,
        branch_examples,
        rng,
        branch_position,
        batch_size,
        terminator,
    )


def train_branch_binding_fallback(
    model: Any,
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    params: list[Scalar] | None = None,
) -> float:
    return train_direct_answer_lesson(
        model,
        fallback_lesson,
        rng,
        learning_rate,
        params=params,
    )


def branch_binding_batch_or_fallback(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    branch_position: int,
    batch_size: int,
    terminator: str,
    params: list[Scalar] | None,
    balance_targets: bool,
) -> list[Any] | float:
    branches = branch_binding_batch(
        model,
        tokenizer,
        example,
        branch_examples,
        rng,
        branch_position,
        batch_size,
        terminator,
        balance_targets,
    )
    if branches:
        return branches
    return train_branch_binding_fallback(
        model,
        fallback_lesson,
        rng,
        learning_rate,
        params,
    )
