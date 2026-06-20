"""Retention-anchored profile-balanced branch objectives."""

from __future__ import annotations

import random
from typing import Any

from answer_model import AnswerExample
from autograd import Scalar
from tokenizer import CharTokenizer
from transformer_direct_answer_core import DirectAnswerLesson
from transformer_direct_answer_profile_balanced_batches import (
    direct_answer_profile_balanced_branch_batch,
    unprofiled_branch_records,
)
from transformer_direct_answer_repairs import train_direct_answer_lesson
from transformer_direct_modes import ANSWER_TERMINATOR
from transformer_profile_balanced_retention_anchors import (
    profile_balanced_retention_anchor_batch,
)


def train_direct_answer_profile_balanced_retention_rank_margin_unlikelihood(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    margin_weight: float,
    branch_position: int,
    batch_size: int,
    hard_negative_count: int,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    branches = direct_answer_profile_balanced_branch_batch(
        model,
        tokenizer,
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

    retention_anchors = profile_balanced_retention_anchor_batch(
        model,
        tokenizer,
        branch_examples,
        rng,
        branch_position,
        batch_size,
        terminator,
    )
    return model.train_step_with_branch_retention_rank_margin(
        unprofiled_branch_records(branches),
        retention_anchors,
        learning_rate,
        negative_weight,
        positive_weight,
        margin_weight,
        hard_negative_count,
        params=params,
    )
