"""Candidate preparation for missing-first-token memory consolidation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from replay_plan import BranchReplayRecord
from transformer_memory_plan_helpers import MissingFirstTokenTargetPlan


@dataclass(frozen=True)
class MissingFirstTokenCandidate:
    target_plan: MissingFirstTokenTargetPlan
    missing_batch: list[BranchReplayRecord]
    model_payload: dict[str, Any]
    optimizer_payload: dict[str, Any]
    records: int


def prepare_missing_first_token_candidate(
    profile: str,
    target_profiles: Sequence[str],
    missing_first_token_ids_by_profile: dict[str, list[int]],
    profile_specific: bool,
    profile_batch: list[BranchReplayRecord],
    rng: random.Random,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    plan_targets: Callable[..., MissingFirstTokenTargetPlan],
    select_anchor_batch: Callable[
        [list[BranchReplayRecord], set[int], random.Random, int],
        list[BranchReplayRecord],
    ],
) -> MissingFirstTokenCandidate:
    target_plan = plan_targets(
        profile,
        target_profiles,
        missing_first_token_ids_by_profile,
        profile_specific=profile_specific,
    )
    missing_batch = select_anchor_batch(
        profile_batch,
        target_plan.target_id_set,
        rng,
        len(profile_batch),
    )
    if not missing_batch:
        return MissingFirstTokenCandidate(
            target_plan=target_plan,
            missing_batch=[],
            model_payload={},
            optimizer_payload={},
            records=0,
        )
    return MissingFirstTokenCandidate(
        target_plan=target_plan,
        missing_batch=missing_batch,
        model_payload=model.to_dict(tokenizer),
        optimizer_payload=optimizer.to_dict(),
        records=len(missing_batch),
    )
