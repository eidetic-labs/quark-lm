"""Representation and output binding objective update helpers."""

from __future__ import annotations

import random
from typing import Any

from answer_model import AnswerExample
from autograd import Scalar
from tokenizer import CharTokenizer
from transformer_direct_answer_branch_binding_shared import (
    branch_binding_batch_or_fallback,
)
from transformer_direct_answer_core import DirectAnswerLesson
from transformer_direct_modes import ANSWER_TERMINATOR


def train_direct_answer_branch_representation_contrast_unlikelihood(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    representation_weight: float,
    branch_position: int,
    batch_size: int,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
    balance_targets: bool = False,
) -> float:
    branches = branch_binding_batch_or_fallback(
        model,
        tokenizer,
        example,
        branch_examples,
        fallback_lesson,
        rng,
        learning_rate,
        branch_position,
        batch_size,
        terminator,
        params,
        balance_targets,
    )
    if isinstance(branches, float):
        return branches
    return model.train_step_with_branch_representation_contrast(
        branches,
        learning_rate,
        negative_weight,
        positive_weight,
        representation_weight,
        params=params,
    )


def train_direct_answer_branch_output_binding_unlikelihood(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    binding_weight: float,
    branch_position: int,
    batch_size: int,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    branches = branch_binding_batch_or_fallback(
        model,
        tokenizer,
        example,
        branch_examples,
        fallback_lesson,
        rng,
        learning_rate,
        branch_position,
        batch_size,
        terminator,
        params,
        False,
    )
    if isinstance(branches, float):
        return branches
    return model.train_step_with_branch_output_binding(
        branches,
        learning_rate,
        negative_weight,
        positive_weight,
        binding_weight,
        params=params,
    )
