"""Target-floor evidence assembly for top-k routing repair."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from replay_plan import BranchReplayRecord
from transformer_branch_replay_competitor_diagnostics import (
    branch_replay_competitor_summary,
)
from transformer_branch_replay_rank_diagnostics import branch_replay_rank_summary
from transformer_profile_balanced_target_floor_anchors import (
    profile_balanced_target_floor_anchors_from_examples,
)


@dataclass(frozen=True)
class RoutingRepairTargetFloorEvidence:
    anchors: list[BranchReplayRecord]
    rank_summary: dict[str, Any]
    competitor_summary: dict[str, Any]


def routing_repair_target_floor_evidence(
    *,
    model: Any,
    tokenizer: Any,
    branch_examples: list[Any],
    rng: random.Random,
    branch_position: int,
    batch_size: int,
    terminator: str,
    repair_target_profiles: list[str],
) -> RoutingRepairTargetFloorEvidence:
    """Build target-floor anchors and their rank/competitor diagnostics."""

    anchors = profile_balanced_target_floor_anchors_from_examples(
        model,
        tokenizer,
        branch_examples,
        rng,
        branch_position,
        batch_size,
        terminator,
        repair_target_profiles=repair_target_profiles,
    )
    return RoutingRepairTargetFloorEvidence(
        anchors=anchors,
        rank_summary=branch_replay_rank_summary(model, anchors),
        competitor_summary=branch_replay_competitor_summary(
            model,
            tokenizer,
            anchors,
        ),
    )
