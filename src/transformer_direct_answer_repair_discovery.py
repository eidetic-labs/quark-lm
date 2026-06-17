"""Direct-answer repair discovery strategies."""

from __future__ import annotations

from typing import Any

from answer_model import AnswerExample
from neural_char_ops import make_context
from tokenizer import CharTokenizer
from transformer_direct_answer_core import (
    DirectAnswerLesson,
    DirectAnswerRepair,
    answer_completion_text,
)
from transformer_direct_modes import ANSWER_TERMINATOR


def direct_answer_first_error(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    terminator: str = ANSWER_TERMINATOR,
) -> tuple[list[int], int, int, int] | None:
    ids = tokenizer.encode(example.prompt)
    target_ids = tokenizer.encode(answer_completion_text(example.target, terminator))
    for position, target_id in enumerate(target_ids):
        context = make_context(ids, model.config.context_size, tokenizer.pad_id)
        probs = model.predict(context)
        predicted_id = max(range(len(probs)), key=lambda index: probs[index])
        if predicted_id != target_id:
            return context, target_id, predicted_id, position
        ids.append(target_id)
    return None


def direct_answer_rollout_error(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    terminator: str = ANSWER_TERMINATOR,
) -> tuple[list[int], int, int, int] | None:
    ids = tokenizer.encode(example.prompt)
    target_ids = tokenizer.encode(answer_completion_text(example.target, terminator))
    last_repair: tuple[list[int], int, int, int] | None = None
    for position, target_id in enumerate(target_ids):
        context = make_context(ids, model.config.context_size, tokenizer.pad_id)
        probs = model.predict(context)
        predicted_id = max(range(len(probs)), key=lambda index: probs[index])
        if predicted_id != target_id:
            last_repair = (context, target_id, predicted_id, position)
        ids.append(predicted_id)
        if terminator and tokenizer.itos[predicted_id] == terminator:
            break
    return last_repair


def direct_answer_early_stop_error(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    terminator: str = ANSWER_TERMINATOR,
) -> tuple[list[int], int, int, int] | None:
    if not terminator:
        return None
    terminator_id = tokenizer.stoi.get(terminator)
    if terminator_id is None:
        return None
    ids = tokenizer.encode(example.prompt)
    target_ids = tokenizer.encode(answer_completion_text(example.target, terminator))
    for position, target_id in enumerate(target_ids):
        context = make_context(ids, model.config.context_size, tokenizer.pad_id)
        probs = model.predict(context)
        predicted_id = max(range(len(probs)), key=lambda index: probs[index])
        if predicted_id == terminator_id and target_id != terminator_id:
            return context, target_id, predicted_id, position
        ids.append(predicted_id)
        if predicted_id == terminator_id:
            break
    return None


def has_repeated_suffix(
    ids: list[int],
    max_ngram_size: int = 3,
    repeat_count: int = 2,
) -> bool:
    if repeat_count < 2:
        return False
    max_size = min(max_ngram_size, len(ids) // repeat_count)
    for ngram_size in range(1, max_size + 1):
        suffix = ids[-ngram_size:]
        repeated = True
        for repeat_index in range(2, repeat_count + 1):
            start = -ngram_size * repeat_index
            end = -ngram_size * (repeat_index - 1)
            if ids[start:end] != suffix:
                repeated = False
                break
        if repeated:
            return True
    return False


def direct_answer_repeat_loop_error(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    terminator: str = ANSWER_TERMINATOR,
) -> tuple[list[int], int, int, int] | None:
    ids = tokenizer.encode(example.prompt)
    generated: list[int] = []
    target_ids = tokenizer.encode(answer_completion_text(example.target, terminator))
    for position, target_id in enumerate(target_ids):
        context = make_context(ids, model.config.context_size, tokenizer.pad_id)
        probs = model.predict(context)
        predicted_id = max(range(len(probs)), key=lambda index: probs[index])
        next_generated = generated + [predicted_id]
        if predicted_id != target_id and has_repeated_suffix(next_generated):
            return context, target_id, predicted_id, position
        ids.append(predicted_id)
        generated = next_generated
        if terminator and tokenizer.itos[predicted_id] == terminator:
            break
    return None


def direct_answer_generated_prefix_recovery(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    recovery_steps: int,
    terminator: str = ANSWER_TERMINATOR,
) -> tuple[list[int], int, int, int, DirectAnswerLesson] | None:
    ids = tokenizer.encode(example.prompt)
    target_ids = tokenizer.encode(answer_completion_text(example.target, terminator))
    for position, target_id in enumerate(target_ids):
        context = make_context(ids, model.config.context_size, tokenizer.pad_id)
        probs = model.predict(context)
        predicted_id = max(range(len(probs)), key=lambda index: probs[index])
        if predicted_id != target_id:
            recovery: DirectAnswerLesson = []
            recovery_ids = ids + [predicted_id]
            for offset in range(max(1, recovery_steps)):
                target_position = position + offset
                if target_position >= len(target_ids):
                    break
                recovery.append(
                    (
                        make_context(
                            recovery_ids,
                            model.config.context_size,
                            tokenizer.pad_id,
                        ),
                        target_ids[target_position],
                    )
                )
                recovery_ids.append(target_ids[target_position])
            if recovery:
                return context, target_id, predicted_id, position, recovery
            return None
        ids.append(predicted_id)
        if terminator and tokenizer.itos[predicted_id] == terminator:
            break
    return None


def direct_answer_sequence_repair_errors(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    terminator: str = ANSWER_TERMINATOR,
) -> list[DirectAnswerRepair]:
    ids = tokenizer.encode(example.prompt)
    target_ids = tokenizer.encode(answer_completion_text(example.target, terminator))
    repairs: list[DirectAnswerRepair] = []
    for position, target_id in enumerate(target_ids):
        context = make_context(ids, model.config.context_size, tokenizer.pad_id)
        probs = model.predict(context)
        predicted_id = max(range(len(probs)), key=lambda index: probs[index])
        if predicted_id != target_id:
            repairs.append((context, target_id, predicted_id, position))
        ids.append(target_id)
    return repairs
