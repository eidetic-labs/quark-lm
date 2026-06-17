"""Batch-contrast branch objective for direct-answer training."""

from __future__ import annotations

import random
from typing import Any

from answer_model import AnswerExample
from autograd import Scalar
from tokenizer import CharTokenizer
from transformer_direct_answer_batches import direct_answer_branch_batch
from transformer_direct_answer_core import DirectAnswerLesson
from transformer_direct_answer_repairs import train_direct_answer_lesson
from transformer_direct_modes import ANSWER_TERMINATOR


def train_direct_answer_branch_batch_contrast_unlikelihood(
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
    batch_size: int,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    branches = direct_answer_branch_batch(
        model,
        tokenizer,
        example,
        branch_examples,
        rng,
        branch_position,
        batch_size,
        terminator,
    )
    if not branches:
        return train_direct_answer_lesson(
            model,
            fallback_lesson,
            rng,
            learning_rate,
            params=params,
        )
    return model.train_step_with_branch_batch_contrast(
        branches,
        learning_rate,
        negative_weight,
        positive_weight,
        params=params,
    )
