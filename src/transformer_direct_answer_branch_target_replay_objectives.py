"""Target-replay coverage objective update helper."""

from __future__ import annotations

import random
from typing import Any

from answer_model import AnswerExample
from autograd import Scalar
from tokenizer import CharTokenizer
from transformer_direct_answer_branch_binding_shared import (
    branch_binding_batch_or_fallback,
)
from transformer_direct_answer_core import (
    DirectAnswerLesson,
    direct_answer_branch_target_ids,
)
from transformer_direct_modes import ANSWER_TERMINATOR


def train_direct_answer_branch_target_replay_coverage_unlikelihood(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    replay_weight: float,
    branch_position: int,
    batch_size: int,
    hard_negative_count: int,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
    balance_targets: bool = False,
) -> float:
    branches = branch_binding_batch_or_fallback(
        model,
        tokenizer,
        example,
        branch_examples,
        fallback_lesson,
        rng,
        learning_rate,
        branch_position,
        batch_size,
        terminator,
        params,
        balance_targets,
    )
    if isinstance(branches, float):
        return branches
    replay_targets = direct_answer_branch_target_ids(
        model,
        tokenizer,
        branch_examples,
        branch_position,
        terminator,
    )
    return model.train_step_with_branch_target_replay_coverage(
        branches,
        replay_targets,
        learning_rate,
        negative_weight,
        positive_weight,
        replay_weight,
        hard_negative_count,
        params=params,
    )
