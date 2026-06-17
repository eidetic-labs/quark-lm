"""Direct-answer hard-branch contrast and repair selection."""

from __future__ import annotations

import random
from typing import Any

from answer_model import AnswerExample
from tokenizer import CharTokenizer
from transformer_direct_answer_core import (
    DirectAnswerBranchContrast,
    direct_answer_branch_context,
)
from transformer_direct_answer_repair_discovery import (
    direct_answer_early_stop_error,
    direct_answer_first_error,
    direct_answer_repeat_loop_error,
    direct_answer_rollout_error,
)
from transformer_direct_modes import ANSWER_TERMINATOR


def direct_answer_hard_branch_contrast(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    rng: random.Random,
    branch_position: int,
    hard_negative_count: int,
    terminator: str = ANSWER_TERMINATOR,
) -> DirectAnswerBranchContrast | None:
    branch = direct_answer_branch_context(
        model,
        tokenizer,
        example,
        branch_position,
        terminator,
    )
    if branch is None:
        return None
    context, target_id, _position = branch
    if not branch_examples:
        return None
    candidates = _hard_negative_candidates(
        branch_examples,
        rng,
        hard_negative_count,
    )
    best_contrast = _best_hard_branch_contrast(
        model,
        tokenizer,
        example,
        candidates,
        context,
        target_id,
        branch_position,
        terminator,
    )
    if best_contrast is None:
        return None
    contrast_context, contrast_target = best_contrast
    return context, target_id, contrast_context, contrast_target


def direct_answer_balanced_repair_error(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    terminator: str = ANSWER_TERMINATOR,
) -> tuple[list[int], int, int, int] | None:
    for repair_fn in (
        direct_answer_early_stop_error,
        direct_answer_repeat_loop_error,
        direct_answer_rollout_error,
        direct_answer_first_error,
    ):
        repair = repair_fn(model, tokenizer, example, terminator)
        if repair is not None:
            return repair
    return None


def _hard_negative_candidates(
    branch_examples: list[AnswerExample],
    rng: random.Random,
    hard_negative_count: int,
) -> list[AnswerExample]:
    if hard_negative_count <= 0 or hard_negative_count >= len(branch_examples):
        candidates = branch_examples[:]
        rng.shuffle(candidates)
        return candidates
    return rng.sample(branch_examples, hard_negative_count)


def _best_hard_branch_contrast(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    candidates: list[AnswerExample],
    context: list[int],
    target_id: int,
    branch_position: int,
    terminator: str,
) -> tuple[list[int], int] | None:
    probs = model.predict(context)
    best_score: float | None = None
    best_contrast: tuple[list[int], int] | None = None
    for contrast_example in candidates:
        if contrast_example == example:
            continue
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
        contrast_probs = model.predict(contrast_context)
        score = probs[contrast_target] + contrast_probs[target_id]
        if best_score is None or score > best_score:
            best_score = score
            best_contrast = (contrast_context, contrast_target)
    return best_contrast
