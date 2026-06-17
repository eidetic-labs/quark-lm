"""Branch-position direct-answer repair objectives."""

from __future__ import annotations

import random
from typing import Any

from answer_model import AnswerExample
from autograd import Scalar
from tokenizer import CharTokenizer
from transformer_direct_answer_core import DirectAnswerLesson
from transformer_direct_answer_branch_repairs import (
    direct_answer_branch_repair_error,
    direct_answer_branch_span_repair_error,
)
from transformer_direct_answer_repair_objective_steps import (
    train_sampled_positive_repair,
)
from transformer_direct_modes import ANSWER_TERMINATOR


def train_direct_answer_branch_repair_unlikelihood(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    branch_position: int,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    repair = direct_answer_branch_repair_error(
        model,
        tokenizer,
        example,
        branch_position,
        terminator,
    )
    return train_sampled_positive_repair(
        model,
        repair,
        fallback_lesson,
        rng,
        learning_rate,
        negative_weight,
        positive_weight,
        params=params,
    )


def train_direct_answer_branch_span_repair_unlikelihood(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    branch_position: int,
    branch_span: int,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    repair = direct_answer_branch_span_repair_error(
        model,
        tokenizer,
        example,
        rng,
        branch_position,
        branch_span,
        terminator,
    )
    return train_sampled_positive_repair(
        model,
        repair,
        fallback_lesson,
        rng,
        learning_rate,
        negative_weight,
        positive_weight,
        params=params,
    )
