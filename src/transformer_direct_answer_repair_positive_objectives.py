"""Positive-balanced direct-answer repair objectives."""

from __future__ import annotations

import random
from typing import Any

from answer_model import AnswerExample
from autograd import Scalar
from tokenizer import CharTokenizer
from transformer_direct_answer_core import DirectAnswerLesson
from transformer_direct_answer_repair_objective_steps import (
    sampled_positive_step,
    train_positive_unlikelihood_repair,
    train_sampled_positive_repair,
)
from transformer_direct_answer_repair_selection import direct_answer_balanced_repair_error
from transformer_direct_answer_repair_discovery import (
    direct_answer_repeat_loop_error,
    direct_answer_sequence_repair_errors,
)
from transformer_direct_modes import ANSWER_TERMINATOR


def train_direct_answer_balanced_repair_unlikelihood(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    repair = direct_answer_balanced_repair_error(model, tokenizer, example, terminator)
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


def train_direct_answer_sequence_repair_unlikelihood(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    repairs = direct_answer_sequence_repair_errors(model, tokenizer, example, terminator)
    positive_context, positive_target = sampled_positive_step(fallback_lesson, rng)
    if not repairs:
        return model.train_step(
            positive_context,
            positive_target,
            learning_rate,
            params=params,
        )
    repair = repairs[rng.randrange(len(repairs))]
    return train_positive_unlikelihood_repair(
        model,
        repair,
        positive_context,
        positive_target,
        learning_rate,
        negative_weight,
        positive_weight,
        params=params,
    )


def train_direct_answer_loop_escape_unlikelihood(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    repair = direct_answer_repeat_loop_error(model, tokenizer, example, terminator)
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
