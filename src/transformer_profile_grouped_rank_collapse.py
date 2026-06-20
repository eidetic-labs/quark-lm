"""Profile-grouped rank-collapse training helpers."""

from __future__ import annotations

from typing import Any

from autograd import Scalar
from replay_plan import BranchReplayRecord, branch_replay_parts


def train_profile_grouped_rank_collapse(
    model: Any,
    branches: list[BranchReplayRecord],
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    margin_weight: float,
    collapse_weight: float,
    hard_negative_count: int,
    params: list[Scalar] | None = None,
) -> float:
    """Train rank-collapse pressure independently for each profile family."""

    total_loss = 0.0
    total_records = 0
    for group in profiled_unprofiled_branch_groups(branches):
        loss = model.train_step_with_branch_rank_collapse_margin(
            group,
            learning_rate,
            negative_weight,
            positive_weight,
            margin_weight,
            collapse_weight,
            hard_negative_count,
            params=params,
        )
        total_loss += float(loss) * len(group)
        total_records += len(group)
    return total_loss / max(total_records, 1)


def profiled_unprofiled_branch_groups(
    branches: list[BranchReplayRecord],
) -> list[list[tuple[list[int], int, int]]]:
    groups: dict[str, list[tuple[list[int], int, int]]] = {}
    for branch in branches:
        context, target, predicted, profile = branch_replay_parts(branch)
        groups.setdefault(profile, []).append((context, target, predicted))
    return [groups[profile] for profile in sorted(groups)]
