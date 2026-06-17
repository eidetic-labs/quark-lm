from __future__ import annotations

import random

from support.core import ANSWER_TERMINATOR, AnswerExample, CharTokenizer, TinyTransformerLM
from support.direct_answer import (
    direct_answer_lesson,
    train_direct_answer_branch_target_diversity_unlikelihood,
    train_direct_answer_branch_target_replay_coverage_unlikelihood,
    train_direct_answer_branch_target_set_coverage_unlikelihood,
)


def direct_answer_training_lesson(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    lesson_example: AnswerExample,
) -> dict[str, object]:
    return direct_answer_lesson(
        tokenizer,
        model.config.context_size,
        lesson_example,
        ANSWER_TERMINATOR,
    )


def train_target_set_coverage_steps(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    lesson_example: AnswerExample,
    examples: list[AnswerExample],
    *,
    repeat: int,
    rng_seed: int,
) -> None:
    lesson = direct_answer_training_lesson(model, tokenizer, lesson_example)
    rng = random.Random(rng_seed)
    for _ in range(repeat):
        train_direct_answer_branch_target_set_coverage_unlikelihood(
            model,
            tokenizer,
            lesson_example,
            examples,
            lesson,
            rng,
            learning_rate=0.03,
            negative_weight=1.0,
            positive_weight=0.0,
            coverage_weight=2.0,
            branch_position=1,
            batch_size=3,
            hard_negative_count=5,
            terminator=ANSWER_TERMINATOR,
            balance_targets=True,
        )


def train_target_diversity_steps(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    lesson_example: AnswerExample,
    examples: list[AnswerExample],
    *,
    repeat: int,
    rng_seed: int,
) -> None:
    lesson = direct_answer_training_lesson(model, tokenizer, lesson_example)
    rng = random.Random(rng_seed)
    for _ in range(repeat):
        train_direct_answer_branch_target_diversity_unlikelihood(
            model,
            tokenizer,
            lesson_example,
            examples,
            lesson,
            rng,
            learning_rate=0.03,
            negative_weight=1.0,
            positive_weight=0.0,
            diversity_weight=2.0,
            branch_position=1,
            batch_size=3,
            hard_negative_count=5,
            terminator=ANSWER_TERMINATOR,
            balance_targets=True,
        )


def train_target_replay_coverage_steps(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    lesson_example: AnswerExample,
    examples: list[AnswerExample],
    *,
    repeat: int,
    rng_seed: int,
) -> None:
    lesson = direct_answer_training_lesson(model, tokenizer, lesson_example)
    rng = random.Random(rng_seed)
    for _ in range(repeat):
        train_direct_answer_branch_target_replay_coverage_unlikelihood(
            model,
            tokenizer,
            lesson_example,
            examples,
            lesson,
            rng,
            learning_rate=0.03,
            negative_weight=1.0,
            positive_weight=0.0,
            replay_weight=2.0,
            branch_position=1,
            batch_size=2,
            hard_negative_count=5,
            terminator=ANSWER_TERMINATOR,
            balance_targets=True,
        )
