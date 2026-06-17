"""Branch-batch construction helpers for direct-answer training."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

from answer_model import AnswerExample
from transformer_direct_answer_profiled_batches import (
    direct_answer_profiled_branch_batch,
    direct_answer_profiled_replay_records,
)
from tokenizer import CharTokenizer
from transformer_direct_answer_core import direct_answer_branch_context
from transformer_direct_modes import ANSWER_TERMINATOR


def direct_answer_dominant_branch_prediction(
    model: Any,
    tokenizer: CharTokenizer,
    branch_examples: list[AnswerExample],
    rng: random.Random,
    branch_position: int,
    sample_count: int,
    terminator: str = ANSWER_TERMINATOR,
) -> tuple[int, int, int] | None:
    if not branch_examples:
        return None
    if sample_count <= 0 or sample_count >= len(branch_examples):
        candidates = branch_examples[:]
        rng.shuffle(candidates)
    else:
        candidates = rng.sample(branch_examples, sample_count)
    predicted_counts: Counter[int] = Counter()
    scored = 0
    for candidate in candidates:
        branch = direct_answer_branch_context(
            model,
            tokenizer,
            candidate,
            branch_position,
            terminator,
        )
        if branch is None:
            continue
        context, _target_id, _position = branch
        probs = model.predict(context)
        predicted_id = max(range(len(probs)), key=lambda index: probs[index])
        predicted_counts[predicted_id] += 1
        scored += 1
    if not predicted_counts:
        return None
    predicted_id, count = predicted_counts.most_common(1)[0]
    return predicted_id, count, scored


def direct_answer_branch_batch(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    rng: random.Random,
    branch_position: int,
    batch_size: int,
    terminator: str = ANSWER_TERMINATOR,
) -> list[tuple[list[int], int]]:
    branch = direct_answer_branch_context(
        model,
        tokenizer,
        example,
        branch_position,
        terminator,
    )
    if branch is None:
        return []
    context, target_id, _position = branch
    branches = [(context, target_id)]
    seen_targets = {target_id}
    candidates = branch_examples[:]
    rng.shuffle(candidates)
    for candidate in candidates:
        if len(branches) >= max(1, batch_size):
            break
        candidate_branch = direct_answer_branch_context(
            model,
            tokenizer,
            candidate,
            branch_position,
            terminator,
        )
        if candidate_branch is None:
            continue
        candidate_context, candidate_target, _candidate_position = candidate_branch
        if candidate_target in seen_targets:
            continue
        branches.append((candidate_context, candidate_target))
        seen_targets.add(candidate_target)
    return branches


def direct_answer_target_balanced_branch_batch(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    rng: random.Random,
    branch_position: int,
    batch_size: int,
    terminator: str = ANSWER_TERMINATOR,
) -> list[tuple[list[int], int]]:
    branch = direct_answer_branch_context(
        model,
        tokenizer,
        example,
        branch_position,
        terminator,
    )
    if branch is None:
        return []
    context, target_id, _position = branch
    branches = [(context, target_id)]
    by_target: dict[int, list[tuple[list[int], int]]] = {}
    candidates = branch_examples[:]
    rng.shuffle(candidates)
    for candidate in candidates:
        candidate_branch = direct_answer_branch_context(
            model,
            tokenizer,
            candidate,
            branch_position,
            terminator,
        )
        if candidate_branch is None:
            continue
        candidate_context, candidate_target, _candidate_position = candidate_branch
        if candidate_target == target_id:
            continue
        by_target.setdefault(candidate_target, []).append(
            (candidate_context, candidate_target)
        )
    target_ids = list(by_target)
    rng.shuffle(target_ids)
    for candidate_target in target_ids:
        if len(branches) >= max(1, batch_size):
            break
        branches.append(rng.choice(by_target[candidate_target]))
    return branches


def direct_answer_branch_diversity_batch(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    rng: random.Random,
    branch_position: int,
    batch_size: int,
    terminator: str = ANSWER_TERMINATOR,
) -> list[tuple[list[int], int, int]]:
    branches = direct_answer_branch_batch(
        model,
        tokenizer,
        example,
        branch_examples,
        rng,
        branch_position,
        batch_size,
        terminator,
    )
    diversity_branches: list[tuple[list[int], int, int]] = []
    for context, target_id in branches:
        probs = model.predict(context)
        predicted_id = max(range(len(probs)), key=lambda index: probs[index])
        diversity_branches.append((context, target_id, predicted_id))
    return diversity_branches


def direct_answer_target_balanced_branch_diversity_batch(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    rng: random.Random,
    branch_position: int,
    batch_size: int,
    terminator: str = ANSWER_TERMINATOR,
) -> list[tuple[list[int], int, int]]:
    branches = direct_answer_target_balanced_branch_batch(
        model,
        tokenizer,
        example,
        branch_examples,
        rng,
        branch_position,
        batch_size,
        terminator,
    )
    diversity_branches: list[tuple[list[int], int, int]] = []
    for context, target_id in branches:
        probs = model.predict(context)
        predicted_id = max(range(len(probs)), key=lambda index: probs[index])
        diversity_branches.append((context, target_id, predicted_id))
    return diversity_branches
