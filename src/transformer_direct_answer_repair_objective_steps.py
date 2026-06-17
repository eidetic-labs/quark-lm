"""Shared update steps for direct-answer repair objectives."""

from __future__ import annotations

import random
from typing import Any

from autograd import Scalar
from transformer_direct_answer_core import DirectAnswerLesson, DirectAnswerRepair


def sampled_positive_step(
    lesson: DirectAnswerLesson,
    rng: random.Random,
) -> tuple[list[int], int]:
    return lesson[rng.randrange(len(lesson))]


def train_unlikelihood_repair(
    model: Any,
    repair: DirectAnswerRepair,
    learning_rate: float,
    negative_weight: float,
    params: list[Scalar] | None = None,
) -> float:
    context, target_id, predicted_id, _position = repair
    return model.train_step_with_unlikelihood(
        context,
        target_id,
        predicted_id,
        learning_rate,
        negative_weight,
        params=params,
    )


def train_positive_unlikelihood_repair(
    model: Any,
    repair: DirectAnswerRepair,
    positive_context: list[int],
    positive_target: int,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    params: list[Scalar] | None = None,
) -> float:
    context, target_id, predicted_id, _position = repair
    return model.train_step_with_unlikelihood_and_positive(
        context,
        target_id,
        predicted_id,
        positive_context,
        positive_target,
        learning_rate,
        negative_weight,
        positive_weight,
        params=params,
    )


def train_sampled_positive_repair(
    model: Any,
    repair: DirectAnswerRepair | None,
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    params: list[Scalar] | None = None,
) -> float:
    positive_context, positive_target = sampled_positive_step(fallback_lesson, rng)
    if repair is None:
        return model.train_step(
            positive_context,
            positive_target,
            learning_rate,
            params=params,
        )
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
