"""Retention anchors for profile-balanced routing repair."""

from __future__ import annotations

import random
from typing import Any

from answer_model import AnswerExample
from replay_plan import BranchReplayRecord
from tokenizer import CharTokenizer
from transformer_baseline_floor_anchor_batches import (
    baseline_floor_objective_anchor_batch,
    baseline_floor_repair_anchor_records,
)
from transformer_direct_answer_core import direct_answer_branch_context
from transformer_direct_answer_profile_keys import direct_answer_training_profile_key
from transformer_direct_modes import ANSWER_TERMINATOR


def profile_balanced_retention_anchor_batch(
    model: Any,
    tokenizer: CharTokenizer,
    branch_examples: list[AnswerExample],
    rng: random.Random,
    branch_position: int,
    batch_size: int,
    terminator: str = ANSWER_TERMINATOR,
) -> list[BranchReplayRecord]:
    """Sample training-family anchors that preserve already represented targets."""

    anchors = baseline_floor_repair_anchor_records(
        _profile_balanced_replay_records(
            model,
            tokenizer,
            branch_examples,
            branch_position,
            terminator,
        )
    )
    return baseline_floor_objective_anchor_batch(
        anchors,
        rng,
        max(1, batch_size),
    )


def _profile_balanced_replay_records(
    model: Any,
    tokenizer: CharTokenizer,
    branch_examples: list[AnswerExample],
    branch_position: int,
    terminator: str,
) -> list[BranchReplayRecord]:
    records: list[BranchReplayRecord] = []
    seen_examples: set[tuple[str, str, str]] = set()
    for example in branch_examples:
        example_key = (example.prompt, example.target, example.source)
        if example_key in seen_examples:
            continue
        seen_examples.add(example_key)
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
                direct_answer_training_profile_key(example),
            )
        )
    return records


def _predicted_id(model: Any, context: list[int]) -> int:
    probs = model.predict(context)
    return max(range(len(probs)), key=lambda index: probs[index])
