"""Pairwise branch contrast objective update helper."""

from __future__ import annotations

import random
from typing import Any

from answer_model import AnswerExample
from autograd import Scalar
from tokenizer import CharTokenizer
from transformer_direct_answer_core import (
    DirectAnswerLesson,
    direct_answer_branch_context,
)
from transformer_direct_answer_repair_objectives import (
    train_direct_answer_branch_repair_unlikelihood,
)
from transformer_direct_answer_repairs import train_direct_answer_lesson
from transformer_direct_modes import ANSWER_TERMINATOR


def train_direct_answer_branch_contrast_unlikelihood(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    contrast_weight: float,
    branch_position: int,
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
    for _ in range(max(len(branch_examples), 1)):
        contrast_example = branch_examples[rng.randrange(len(branch_examples))]
        contrast = direct_answer_branch_context(
            model,
            tokenizer,
            contrast_example,
            branch_position,
            terminator,
        )
        if contrast is None:
            continue
        contrast_context, contrast_target, _contrast_position = contrast
        if contrast_target == target_id:
            continue
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
    return train_direct_answer_branch_repair_unlikelihood(
        model,
        tokenizer,
        example,
        fallback_lesson,
        rng,
        learning_rate,
        negative_weight,
        contrast_weight,
        branch_position,
        terminator,
        params=params,
    )
