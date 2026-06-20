"""Target-floor anchors for profile-balanced repair branches."""

from __future__ import annotations

import random
from typing import Any

from answer_model import AnswerExample
from replay_plan import BranchReplayRecord, branch_replay_parts
from tokenizer import CharTokenizer
from transformer_baseline_floor_anchor_batches import (
    baseline_floor_objective_anchor_batch,
)
from transformer_direct_answer_profile_balanced_batches import (
    direct_answer_profile_balanced_branch_batch,
)
from transformer_direct_modes import ANSWER_TERMINATOR


def profile_balanced_target_floor_anchor_batch(
    branches: list[BranchReplayRecord],
    rng: random.Random,
    batch_size: int,
) -> list[BranchReplayRecord]:
    """Sample balanced anchors that lift declared repair targets directly."""

    anchors = []
    seen: set[tuple[tuple[int, ...], int, str]] = set()
    for branch in branches:
        context, target, _predicted, profile = branch_replay_parts(branch)
        key = (tuple(context), target, profile)
        if key in seen:
            continue
        seen.add(key)
        anchors.append((context, target, target, profile))
    return baseline_floor_objective_anchor_batch(anchors, rng, max(1, batch_size))


def profile_balanced_target_floor_anchors_from_examples(
    model: Any,
    tokenizer: CharTokenizer,
    branch_examples: list[AnswerExample],
    rng: random.Random,
    branch_position: int,
    batch_size: int,
    terminator: str = ANSWER_TERMINATOR,
    repair_target_profiles: list[str] | tuple[str, ...] | None = None,
) -> list[BranchReplayRecord]:
    """Sample target-floor anchors across all target tokens in repair profiles."""

    floor_branches = direct_answer_profile_balanced_branch_batch(
        model,
        tokenizer,
        branch_examples,
        rng,
        branch_position,
        max(batch_size, len(branch_examples)),
        terminator,
        repair_target_profiles=repair_target_profiles,
    )
    return profile_balanced_target_floor_anchor_batch(
        floor_branches,
        rng,
        max(batch_size, len(floor_branches)),
    )
