"""Branch ranking and top-k objective update helpers."""

from __future__ import annotations

import random
from typing import Any

from answer_model import AnswerExample
from autograd import Scalar
from tokenizer import CharTokenizer
from transformer_direct_answer_batches import (
    direct_answer_branch_diversity_batch,
    direct_answer_target_balanced_branch_diversity_batch,
)
from transformer_direct_answer_core import DirectAnswerLesson
from transformer_direct_answer_profile_balanced_ranking_objectives import (
    train_direct_answer_profile_balanced_branch_rank_collapse_unlikelihood,
    train_direct_answer_profile_balanced_branch_rank_margin_unlikelihood,
    train_direct_answer_profile_balanced_branch_topk_softmax_unlikelihood,
)
from transformer_direct_answer_repairs import train_direct_answer_lesson
from transformer_direct_modes import ANSWER_TERMINATOR


def train_direct_answer_branch_rank_margin_unlikelihood(
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
    balance_targets: bool = False,
) -> float:
    branches = _branch_diversity_batch(
        model,
        tokenizer,
        example,
        branch_examples,
        rng,
        branch_position,
        batch_size,
        terminator,
        balance_targets,
    )
    if not branches:
        return train_direct_answer_lesson(
            model,
            fallback_lesson,
            rng,
            learning_rate,
            params=params,
        )
    return model.train_step_with_branch_rank_margin(
        branches,
        learning_rate,
        negative_weight,
        positive_weight,
        margin_weight,
        hard_negative_count,
        params=params,
    )


def train_direct_answer_branch_topk_softmax_unlikelihood(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    candidate_weight: float,
    branch_position: int,
    batch_size: int,
    candidate_count: int,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
    balance_targets: bool = False,
) -> float:
    branches = _branch_diversity_batch(
        model,
        tokenizer,
        example,
        branch_examples,
        rng,
        branch_position,
        batch_size,
        terminator,
        balance_targets,
    )
    if not branches:
        return train_direct_answer_lesson(
            model,
            fallback_lesson,
            rng,
            learning_rate,
            params=params,
        )
    return model.train_step_with_branch_topk_softmax(
        branches,
        learning_rate,
        negative_weight,
        positive_weight,
        candidate_weight,
        candidate_count,
        params=params,
    )

def _branch_diversity_batch(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    rng: random.Random,
    branch_position: int,
    batch_size: int,
    terminator: str,
    balance_targets: bool,
) -> list[tuple[list[int], int, int]]:
    batch_builder = (
        direct_answer_target_balanced_branch_diversity_batch
        if balance_targets
        else direct_answer_branch_diversity_batch
    )
    return batch_builder(
        model,
        tokenizer,
        example,
        branch_examples,
        rng,
        branch_position,
        batch_size,
        terminator,
    )
