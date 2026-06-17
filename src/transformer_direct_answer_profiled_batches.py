"""Profiled branch replay batch construction for direct-answer training."""

from __future__ import annotations

import random
from typing import Any

from answer_model import AnswerExample
from replay_plan import (
    BranchReplayRecord,
    ProfiledBranchSeed,
    direct_answer_profile_key,
)
from tokenizer import CharTokenizer
from transformer_direct_answer_core import direct_answer_branch_context
from transformer_direct_modes import ANSWER_TERMINATOR, ReplayPredictionOverrides


def direct_answer_profiled_branch_batch(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    rng: random.Random,
    branch_position: int,
    batch_size: int,
    terminator: str = ANSWER_TERMINATOR,
    balance_targets: bool = False,
    prediction_overrides: ReplayPredictionOverrides | None = None,
) -> list[BranchReplayRecord]:
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
    seeds: list[ProfiledBranchSeed] = [
        (context, target_id, direct_answer_profile_key(example))
    ]
    candidates = branch_examples[:]
    rng.shuffle(candidates)
    if balance_targets:
        _add_target_balanced_profiled_seeds(
            seeds,
            model,
            tokenizer,
            candidates,
            rng,
            branch_position,
            batch_size,
            terminator,
            target_id,
        )
    else:
        _add_profiled_seeds(
            seeds,
            model,
            tokenizer,
            candidates,
            branch_position,
            batch_size,
            terminator,
            target_id,
        )
    return _profiled_records(model, seeds, prediction_overrides)


def direct_answer_profiled_replay_records(
    model: Any,
    tokenizer: CharTokenizer,
    branch_examples: list[AnswerExample],
    branch_position: int,
    terminator: str = ANSWER_TERMINATOR,
) -> list[BranchReplayRecord]:
    records: list[BranchReplayRecord] = []
    for example in branch_examples:
        branch = direct_answer_branch_context(
            model,
            tokenizer,
            example,
            branch_position,
            terminator,
        )
        if branch is None:
            continue
        context, target_id, _position = branch
        predicted_id = _predicted_id(model, context)
        records.append(
            (
                context,
                target_id,
                predicted_id,
                direct_answer_profile_key(example),
            )
        )
    return records


def _add_target_balanced_profiled_seeds(
    seeds: list[ProfiledBranchSeed],
    model: Any,
    tokenizer: CharTokenizer,
    candidates: list[AnswerExample],
    rng: random.Random,
    branch_position: int,
    batch_size: int,
    terminator: str,
    target_id: int,
) -> None:
    by_target: dict[int, list[ProfiledBranchSeed]] = {}
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
            (
                candidate_context,
                candidate_target,
                direct_answer_profile_key(candidate),
            )
        )
    target_ids = list(by_target)
    rng.shuffle(target_ids)
    for candidate_target in target_ids:
        if len(seeds) >= max(1, batch_size):
            break
        seeds.append(rng.choice(by_target[candidate_target]))


def _add_profiled_seeds(
    seeds: list[ProfiledBranchSeed],
    model: Any,
    tokenizer: CharTokenizer,
    candidates: list[AnswerExample],
    branch_position: int,
    batch_size: int,
    terminator: str,
    target_id: int,
) -> None:
    seen_targets = {target_id}
    for candidate in candidates:
        if len(seeds) >= max(1, batch_size):
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
        seeds.append(
            (
                candidate_context,
                candidate_target,
                direct_answer_profile_key(candidate),
            )
        )
        seen_targets.add(candidate_target)


def _profiled_records(
    model: Any,
    seeds: list[ProfiledBranchSeed],
    prediction_overrides: ReplayPredictionOverrides | None,
) -> list[BranchReplayRecord]:
    records: list[BranchReplayRecord] = []
    for context, target_id, profile in seeds:
        override_key = (tuple(context), target_id, profile)
        predicted_id = (
            prediction_overrides[override_key]
            if prediction_overrides is not None
            and override_key in prediction_overrides
            else None
        )
        if predicted_id is None:
            predicted_id = _predicted_id(model, context)
        records.append((context, target_id, predicted_id, profile))
    return records


def _predicted_id(model: Any, context: list[int]) -> int:
    probs = model.predict(context)
    return max(range(len(probs)), key=lambda index: probs[index])
