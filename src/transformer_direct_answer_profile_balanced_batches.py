"""Profile-balanced branch batches for direct-answer routing repair."""

from __future__ import annotations

import random
from typing import Any

from answer_model import AnswerExample
from replay_plan import BranchReplayRecord
from tokenizer import CharTokenizer
from transformer_direct_answer_core import direct_answer_branch_context
from transformer_direct_answer_profile_keys import direct_answer_training_profile_key
from transformer_direct_modes import ANSWER_TERMINATOR

ProfiledBranchSeed = tuple[list[int], int, str]


def direct_answer_profile_balanced_branch_batch(
    model: Any,
    tokenizer: CharTokenizer,
    branch_examples: list[AnswerExample],
    rng: random.Random,
    branch_position: int,
    batch_size: int,
    terminator: str = ANSWER_TERMINATOR,
) -> list[BranchReplayRecord]:
    """Build a bounded batch that covers trainable profile families first."""

    grouped = _profile_target_seeds(
        model,
        tokenizer,
        branch_examples,
        branch_position,
        terminator,
    )
    profiles = sorted(grouped)
    rng.shuffle(profiles)
    if not profiles:
        return []

    target_by_profile = {
        profile: sorted(targets)
        for profile, targets in grouped.items()
    }
    for target_ids in target_by_profile.values():
        rng.shuffle(target_ids)

    seeds: list[ProfiledBranchSeed] = []
    selected: set[tuple[str, int]] = set()
    for profile in profiles:
        _append_next_profile_target(
            seeds,
            selected,
            grouped,
            target_by_profile,
            profile,
            rng,
        )

    max_records = max(len(profiles), max(1, batch_size))
    while len(seeds) < max_records:
        progressed = False
        for profile in profiles:
            if len(seeds) >= max_records:
                break
            progressed = _append_next_profile_target(
                seeds,
                selected,
                grouped,
                target_by_profile,
                profile,
                rng,
            ) or progressed
        if not progressed:
            break
    return _profiled_records(model, seeds)


def unprofiled_branch_records(
    branches: list[BranchReplayRecord],
) -> list[tuple[list[int], int, int]]:
    """Strip profile keys for model trainers that operate on branch triples."""

    return [(branch[0], branch[1], branch[2]) for branch in branches]


def _profile_target_seeds(
    model: Any,
    tokenizer: CharTokenizer,
    branch_examples: list[AnswerExample],
    branch_position: int,
    terminator: str,
) -> dict[str, dict[int, list[ProfiledBranchSeed]]]:
    grouped: dict[str, dict[int, list[ProfiledBranchSeed]]] = {}
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
        profile = direct_answer_training_profile_key(example)
        grouped.setdefault(profile, {}).setdefault(target_id, []).append(
            (context, target_id, profile)
        )
    return grouped


def _append_next_profile_target(
    seeds: list[ProfiledBranchSeed],
    selected: set[tuple[str, int]],
    grouped: dict[str, dict[int, list[ProfiledBranchSeed]]],
    target_by_profile: dict[str, list[int]],
    profile: str,
    rng: random.Random,
) -> bool:
    for target_id in target_by_profile.get(profile, []):
        key = (profile, target_id)
        if key in selected:
            continue
        seeds.append(rng.choice(grouped[profile][target_id]))
        selected.add(key)
        return True
    return False


def _profiled_records(
    model: Any,
    seeds: list[ProfiledBranchSeed],
) -> list[BranchReplayRecord]:
    records: list[BranchReplayRecord] = []
    for context, target_id, profile in seeds:
        probs = model.predict(context)
        predicted_id = max(range(len(probs)), key=lambda index: probs[index])
        records.append((context, target_id, predicted_id, profile))
    return records
