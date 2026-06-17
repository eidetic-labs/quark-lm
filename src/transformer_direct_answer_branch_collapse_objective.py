"""Branch-collapse repair objective for direct-answer training."""

from __future__ import annotations

import random
from typing import Any

from answer_model import AnswerExample
from autograd import Scalar
from tokenizer import CharTokenizer
from transformer_direct_answer_batches import direct_answer_dominant_branch_prediction
from transformer_direct_answer_core import (
    DirectAnswerLesson,
    direct_answer_branch_context,
)
from transformer_direct_answer_repairs import train_direct_answer_lesson
from transformer_direct_modes import ANSWER_TERMINATOR


def train_direct_answer_branch_collapse_unlikelihood(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    branch_position: int,
    sample_count: int,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    branch = direct_answer_branch_context(
        model,
        tokenizer,
        example,
        branch_position,
        terminator,
    )
    if branch is None:
        return train_direct_answer_lesson(
            model,
            fallback_lesson,
            rng,
            learning_rate,
            params=params,
    )
    context, target_id, _position = branch
    local_probs = model.predict(context)
    local_predicted_id = max(
        range(len(local_probs)),
        key=lambda index: local_probs[index],
    )
    dominant = direct_answer_dominant_branch_prediction(
        model,
        tokenizer,
        branch_examples,
        rng,
        branch_position,
        sample_count,
        terminator,
    )
    negative_id = local_predicted_id
    if dominant is not None:
        dominant_id, _count, _scored = dominant
        if dominant_id != target_id:
            negative_id = dominant_id
    positive_context, positive_target = fallback_lesson[
        rng.randrange(len(fallback_lesson))
    ]
    return model.train_step_with_unlikelihood_and_positive(
        context,
        target_id,
        negative_id,
        positive_context,
        positive_target,
        learning_rate,
        negative_weight,
        positive_weight,
        params=params,
    )
