"""Compatibility exports and primitive update helpers for direct-answer repairs."""

from __future__ import annotations

import random
from typing import Any

from answer_model import AnswerExample
from autograd import Scalar
from tokenizer import CharTokenizer
from transformer_direct_answer_core import DirectAnswerLesson
from transformer_direct_answer_repair_discovery import direct_answer_first_error
from transformer_direct_modes import ANSWER_TERMINATOR


def train_direct_answer_lesson(
    model: Any,
    lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    params: list[Scalar] | None = None,
) -> float:
    context, target_id = lesson[rng.randrange(len(lesson))]
    return model.train_step(context, target_id, learning_rate, params=params)


def train_direct_answer_first_error(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    repair = direct_answer_first_error(model, tokenizer, example, terminator)
    if repair is None:
        return train_direct_answer_lesson(
            model,
            fallback_lesson,
            rng,
            learning_rate,
            params=params,
        )
    context, target_id, _predicted_id, _position = repair
    return model.train_step(context, target_id, learning_rate, params=params)


def train_direct_answer_first_error_unlikelihood(
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
    repair = direct_answer_first_error(model, tokenizer, example, terminator)
    if repair is None:
        return train_direct_answer_lesson(
            model,
            fallback_lesson,
            rng,
            learning_rate,
            params=params,
        )
    context, target_id, predicted_id, _position = repair
    return model.train_step_with_unlikelihood(
        context,
        target_id,
        predicted_id,
        learning_rate,
        negative_weight,
        params=params,
    )
