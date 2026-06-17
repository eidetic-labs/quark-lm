"""Generated-prefix recovery objectives for direct-answer repairs."""

from __future__ import annotations

import random
from typing import Any

from answer_model import AnswerExample
from autograd import Scalar
from tokenizer import CharTokenizer
from transformer_direct_answer_core import DirectAnswerLesson
from transformer_direct_answer_repair_objective_steps import (
    sampled_positive_step,
    train_positive_unlikelihood_repair,
)
from transformer_direct_answer_repair_positive_objectives import (
    train_direct_answer_balanced_repair_unlikelihood,
)
from transformer_direct_answer_repair_discovery import (
    direct_answer_generated_prefix_recovery,
)
from transformer_direct_modes import ANSWER_TERMINATOR


def train_direct_answer_generated_prefix_recovery_unlikelihood(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    recovery_steps: int,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    repair = direct_answer_generated_prefix_recovery(
        model,
        tokenizer,
        example,
        recovery_steps,
        terminator,
    )
    if repair is None:
        return train_direct_answer_balanced_repair_unlikelihood(
            model,
            tokenizer,
            example,
            fallback_lesson,
            rng,
            learning_rate,
            negative_weight,
            positive_weight,
            terminator,
            params=params,
        )
    context, target_id, predicted_id, position, recovery_lesson = repair
    positive_context, positive_target = sampled_positive_step(recovery_lesson, rng)
    return train_positive_unlikelihood_repair(
        model,
        (context, target_id, predicted_id, position),
        positive_context,
        positive_target,
        learning_rate,
        negative_weight,
        positive_weight,
        params=params,
    )
