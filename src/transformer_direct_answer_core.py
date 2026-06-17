"""Direct-answer sequence and branch-targeting primitives."""

from __future__ import annotations

from typing import Any

from answer_model import AnswerExample
from autograd import Scalar
from neural_char_ops import make_context
from tokenizer import CharTokenizer
from transformer_direct_modes import ANSWER_TERMINATOR
from transformer_math import cross_entropy_scalars


def answer_sequence_nll(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
) -> float:
    prompt_ids = tokenizer.encode(example.prompt)
    target_ids = tokenizer.encode(example.target)
    ids = prompt_ids[:]
    total = 0.0
    for target_id in target_ids:
        context = make_context(ids, model.config.context_size, tokenizer.pad_id)
        total += model.nll(context, target_id)
        ids.append(target_id)
    return total / max(len(target_ids), 1)


def answer_sequence_loss_scalars(
    model: Any,
    tokenizer: CharTokenizer,
    prompt: str,
    target: str,
    max_chars: int = 0,
) -> Scalar:
    prompt_ids = tokenizer.encode(prompt)
    target_ids = tokenizer.encode(target)
    if max_chars > 0:
        target_ids = target_ids[:max_chars]
    ids = prompt_ids[:]
    total = Scalar(0.0)
    for target_id in target_ids:
        context = make_context(ids, model.config.context_size, tokenizer.pad_id)
        total = total + cross_entropy_scalars(model._forward_scalars(context), target_id)
        ids.append(target_id)
    return total / max(len(target_ids), 1)


DirectAnswerLesson = list[tuple[list[int], int]]
DirectAnswerRepair = tuple[list[int], int, int, int]
DirectAnswerBranchContrast = tuple[list[int], int, list[int], int]


def answer_completion_text(target: str, terminator: str = ANSWER_TERMINATOR) -> str:
    return f"{target}{terminator}" if terminator else target


def direct_answer_lesson(
    tokenizer: CharTokenizer,
    context_size: int,
    example: AnswerExample,
    terminator: str = ANSWER_TERMINATOR,
) -> DirectAnswerLesson:
    prompt_ids = tokenizer.encode(example.prompt)
    target_ids = tokenizer.encode(answer_completion_text(example.target, terminator))
    lesson: DirectAnswerLesson = []
    ids = prompt_ids[:]
    for target_id in target_ids:
        lesson.append((make_context(ids, context_size, tokenizer.pad_id), target_id))
        ids.append(target_id)
    return lesson


def direct_answer_sequence_nll(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    terminator: str = ANSWER_TERMINATOR,
) -> float:
    lesson = direct_answer_lesson(tokenizer, model.config.context_size, example, terminator)
    total = 0.0
    for context, target_id in lesson:
        total += model.nll(context, target_id)
    return total / max(len(lesson), 1)


def direct_answer_branch_context(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_position: int,
    terminator: str = ANSWER_TERMINATOR,
) -> tuple[list[int], int, int] | None:
    if branch_position < 0:
        return None
    ids = tokenizer.encode(example.prompt)
    target_ids = tokenizer.encode(answer_completion_text(example.target, terminator))
    if branch_position >= len(target_ids):
        return None
    ids.extend(target_ids[:branch_position])
    context = make_context(ids, model.config.context_size, tokenizer.pad_id)
    return context, target_ids[branch_position], branch_position


def direct_answer_branch_target_ids(
    model: Any,
    tokenizer: CharTokenizer,
    branch_examples: list[AnswerExample],
    branch_position: int,
    terminator: str = ANSWER_TERMINATOR,
) -> list[int]:
    target_ids: set[int] = set()
    for candidate in branch_examples:
        branch = direct_answer_branch_context(
            model,
            tokenizer,
            candidate,
            branch_position,
            terminator,
        )
        if branch is None:
            continue
        _context, target_id, _position = branch
        target_ids.add(target_id)
    return sorted(target_ids)


def direct_answer_branch_span_position(
    tokenizer: CharTokenizer,
    example: AnswerExample,
    rng: random.Random,
    branch_position: int,
    branch_span: int,
    terminator: str = ANSWER_TERMINATOR,
) -> int | None:
    if branch_position < 0:
        return None
    target_ids = tokenizer.encode(answer_completion_text(example.target, terminator))
    if branch_position >= len(target_ids):
        return None
    end_position = min(len(target_ids), branch_position + max(1, branch_span))
    return rng.randrange(branch_position, end_position)
