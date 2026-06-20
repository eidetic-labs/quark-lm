"""Target-focused basic branch objectives for direct-answer training."""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

from answer_model import AnswerExample
from autograd import Scalar
from tokenizer import CharTokenizer
from transformer_direct_answer_batches import direct_answer_branch_diversity_batch
from transformer_direct_answer_core import DirectAnswerLesson
from transformer_direct_answer_profile_balanced_batches import (
    direct_answer_profile_balanced_branch_batch,
    unprofiled_branch_records,
)
from transformer_direct_answer_repairs import train_direct_answer_lesson
from transformer_direct_modes import ANSWER_TERMINATOR


BranchTargetTrainer = Callable[..., float]


def _train_with_diversity_batch_or_lesson(
    *,
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    objective_weight: float,
    branch_position: int,
    batch_size: int,
    terminator: str,
    params: list[Scalar] | None,
    trainer: BranchTargetTrainer,
) -> float:
    branches = direct_answer_branch_diversity_batch(
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
    return trainer(
        branches,
        learning_rate,
        negative_weight,
        positive_weight,
        objective_weight,
        params=params,
    )


def train_direct_answer_branch_target_softmax_unlikelihood(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    target_softmax_weight: float,
    branch_position: int,
    batch_size: int,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    return _train_with_diversity_batch_or_lesson(
        model=model,
        tokenizer=tokenizer,
        example=example,
        branch_examples=branch_examples,
        fallback_lesson=fallback_lesson,
        rng=rng,
        learning_rate=learning_rate,
        negative_weight=negative_weight,
        positive_weight=positive_weight,
        objective_weight=target_softmax_weight,
        branch_position=branch_position,
        batch_size=batch_size,
        terminator=terminator,
        params=params,
        trainer=model.train_step_with_branch_target_softmax,
    )


def train_direct_answer_branch_target_margin_unlikelihood(
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
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    return _train_with_diversity_batch_or_lesson(
        model=model,
        tokenizer=tokenizer,
        example=example,
        branch_examples=branch_examples,
        fallback_lesson=fallback_lesson,
        rng=rng,
        learning_rate=learning_rate,
        negative_weight=negative_weight,
        positive_weight=positive_weight,
        objective_weight=margin_weight,
        branch_position=branch_position,
        batch_size=batch_size,
        terminator=terminator,
        params=params,
        trainer=model.train_step_with_branch_target_margin,
    )


def train_direct_answer_branch_hidden_projection_margin_unlikelihood(
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
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    return _train_with_diversity_batch_or_lesson(
        model=model,
        tokenizer=tokenizer,
        example=example,
        branch_examples=branch_examples,
        fallback_lesson=fallback_lesson,
        rng=rng,
        learning_rate=learning_rate,
        negative_weight=negative_weight,
        positive_weight=positive_weight,
        objective_weight=margin_weight,
        branch_position=branch_position,
        batch_size=batch_size,
        terminator=terminator,
        params=params,
        trainer=model.train_step_with_branch_hidden_projection_margin,
    )


def train_direct_answer_profile_balanced_branch_hidden_projection_margin_unlikelihood(
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
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
    repair_target_profiles: list[str] | None = None,
) -> float:
    branches = direct_answer_profile_balanced_branch_batch(
        model,
        tokenizer,
        branch_examples,
        rng,
        branch_position,
        batch_size,
        terminator,
        repair_target_profiles=repair_target_profiles,
    )
    if not branches:
        return train_direct_answer_lesson(
            model,
            fallback_lesson,
            rng,
            learning_rate,
            params=params,
        )
    return model.train_step_with_branch_hidden_projection_margin(
        unprofiled_branch_records(branches),
        learning_rate,
        negative_weight,
        positive_weight,
        margin_weight,
        params=params,
    )
