"""Negative-only direct-answer repair objectives."""

from __future__ import annotations

import random
from typing import Any

from answer_model import AnswerExample
from autograd import Scalar
from tokenizer import CharTokenizer
from transformer_direct_answer_core import DirectAnswerLesson
from transformer_direct_answer_repair_objective_steps import train_unlikelihood_repair
from transformer_direct_answer_repair_discovery import (
    direct_answer_early_stop_error,
    direct_answer_repeat_loop_error,
    direct_answer_rollout_error,
)
from transformer_direct_answer_repairs import (
    train_direct_answer_first_error_unlikelihood,
    train_direct_answer_lesson,
)
from transformer_direct_modes import ANSWER_TERMINATOR


def train_direct_answer_rollout_unlikelihood(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    repair = direct_answer_rollout_error(model, tokenizer, example, terminator)
    if repair is None:
        return train_direct_answer_lesson(
            model,
            fallback_lesson,
            rng,
            learning_rate,
            params=params,
        )
    return train_unlikelihood_repair(
        model,
        repair,
        learning_rate,
        negative_weight,
        params=params,
    )


def train_direct_answer_repeat_loop_unlikelihood(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    repair = direct_answer_repeat_loop_error(model, tokenizer, example, terminator)
    if repair is None:
        return train_direct_answer_first_error_unlikelihood(
            model,
            tokenizer,
            example,
            fallback_lesson,
            rng,
            learning_rate,
            negative_weight,
            terminator,
            params=params,
        )
    return train_unlikelihood_repair(
        model,
        repair,
        learning_rate,
        negative_weight,
        params=params,
    )


def train_direct_answer_early_stop_unlikelihood(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    repair = direct_answer_early_stop_error(model, tokenizer, example, terminator)
    if repair is None:
        return train_direct_answer_first_error_unlikelihood(
            model,
            tokenizer,
            example,
            fallback_lesson,
            rng,
            learning_rate,
            negative_weight,
            terminator,
            params=params,
        )
    return train_unlikelihood_repair(
        model,
        repair,
        learning_rate,
        negative_weight,
        params=params,
    )
