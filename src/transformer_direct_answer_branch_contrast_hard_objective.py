"""Hard-negative branch contrast objective update helper."""

from __future__ import annotations

import random
from typing import Any

from answer_model import AnswerExample
from autograd import Scalar
from tokenizer import CharTokenizer
from transformer_direct_answer_core import DirectAnswerLesson
from transformer_direct_answer_repair_objectives import (
    train_direct_answer_branch_repair_unlikelihood,
)
from transformer_direct_answer_repair_selection import direct_answer_hard_branch_contrast
from transformer_direct_modes import ANSWER_TERMINATOR


def train_direct_answer_hard_branch_contrast_unlikelihood(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    contrast_weight: float,
    branch_position: int,
    hard_negative_count: int,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    contrast = direct_answer_hard_branch_contrast(
        model,
        tokenizer,
        example,
        branch_examples,
        rng,
        branch_position,
        hard_negative_count,
        terminator,
    )
    if contrast is None:
        return train_direct_answer_branch_repair_unlikelihood(
            model,
            tokenizer,
            example,
            fallback_lesson,
            rng,
            learning_rate,
            negative_weight,
            positive_weight,
            branch_position,
            terminator,
            params=params,
        )
    context, target_id, contrast_context, contrast_target = contrast
    return model.train_step_with_branch_contrast(
        context,
        target_id,
        contrast_context,
        contrast_target,
        learning_rate,
        negative_weight,
        contrast_weight,
        params=params,
    )
