"""Eval-profile retention anchors for guarded routing repair."""

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
from transformer_direct_modes import ANSWER_TERMINATOR


def eval_profile_retention_anchor_batch(
    model: Any,
    tokenizer: CharTokenizer,
    eval_records: dict[str, list[dict[str, Any]]] | None,
    rng: random.Random,
    branch_position: int,
    batch_size: int,
    terminator: str = ANSWER_TERMINATOR,
) -> list[BranchReplayRecord]:
    """Sample eval-profile anchors that preserve represented measured behavior."""

    anchors = baseline_floor_repair_anchor_records(
        eval_profile_replay_records(
            model,
            tokenizer,
            eval_records,
            branch_position,
            terminator,
        )
    )
    return baseline_floor_objective_anchor_batch(anchors, rng, max(1, batch_size))


def eval_profile_replay_records(
    model: Any,
    tokenizer: CharTokenizer,
    eval_records: dict[str, list[dict[str, Any]]] | None,
    branch_position: int,
    terminator: str = ANSWER_TERMINATOR,
) -> list[BranchReplayRecord]:
    """Build replay records from closed-world eval prompts without promoting them."""

    records: list[BranchReplayRecord] = []
    seen_records: set[tuple[str, str, str]] = set()
    for profile, profile_records in sorted((eval_records or {}).items()):
        for record in profile_records:
            prompt = record.get("prompt")
            target = record.get("target")
            if not isinstance(prompt, str) or not isinstance(target, str):
                continue
            record_key = (profile, prompt, target)
            if record_key in seen_records:
                continue
            seen_records.add(record_key)
            branch = direct_answer_branch_context(
                model,
                tokenizer,
                AnswerExample(prompt=prompt, target=target, source=profile),
                branch_position,
                terminator,
            )
            if branch is None:
                continue
            context, target_id, _position = branch
            records.append((context, target_id, _predicted_id(model, context), profile))
    return records


def _predicted_id(model: Any, context: list[int]) -> int:
    probs = model.predict(context)
    return max(range(len(probs)), key=lambda index: probs[index])
