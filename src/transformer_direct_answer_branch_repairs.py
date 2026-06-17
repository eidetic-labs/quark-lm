"""Branch-position repair discovery for direct-answer training."""

from __future__ import annotations

import random
from typing import Any

from answer_model import AnswerExample
from tokenizer import CharTokenizer
from transformer_direct_answer_core import (
    DirectAnswerRepair,
    direct_answer_branch_context,
    direct_answer_branch_span_position,
)
from transformer_direct_modes import ANSWER_TERMINATOR


def direct_answer_branch_repair_error(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_position: int,
    terminator: str = ANSWER_TERMINATOR,
) -> DirectAnswerRepair | None:
    branch = direct_answer_branch_context(
        model,
        tokenizer,
        example,
        branch_position,
        terminator,
    )
    if branch is None:
        return None
    context, target_id, position = branch
    probs = model.predict(context)
    predicted_id = max(range(len(probs)), key=lambda index: probs[index])
    return context, target_id, predicted_id, position


def direct_answer_branch_span_repair_error(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    rng: random.Random,
    branch_position: int,
    branch_span: int,
    terminator: str = ANSWER_TERMINATOR,
) -> DirectAnswerRepair | None:
    sampled_position = direct_answer_branch_span_position(
        tokenizer,
        example,
        rng,
        branch_position,
        branch_span,
        terminator,
    )
    if sampled_position is None:
        return None
    return direct_answer_branch_repair_error(
        model,
        tokenizer,
        example,
        sampled_position,
        terminator,
    )
