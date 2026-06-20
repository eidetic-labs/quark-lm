"""Hidden representation pressure for routing-repair target floors."""

from __future__ import annotations

from typing import Any

from autograd import Scalar
from replay_plan import BranchReplayRecord, branch_replay_parts
from transformer_lm_hidden_contrast import pairwise_hidden_contrast_loss

TopKBranchRecord = tuple[list[int], int, int]
REPRESENTATION_RECORD_LIMIT = 12


def target_floor_representation_loss(
    model: Any,
    branches: list[TopKBranchRecord],
    target_floor_anchors: list[BranchReplayRecord],
    representation_weight: float,
) -> Scalar | None:
    """Separate hidden states for different target tokens in repair batches."""

    if representation_weight <= 0.0:
        return None
    hidden_by_target = _hidden_targets(model, branches, target_floor_anchors)
    if len({target for _hidden, target in hidden_by_target}) < 2:
        return None
    return (
        pairwise_hidden_contrast_loss(
            hidden_by_target,
            model.config.embedding_dim,
        )
        * representation_weight
    )


def _hidden_targets(
    model: Any,
    branches: list[TopKBranchRecord],
    target_floor_anchors: list[BranchReplayRecord],
) -> list[tuple[list[Scalar], int]]:
    hidden_by_target: list[tuple[list[Scalar], int]] = []
    for context, target, _predicted in branches:
        if len(hidden_by_target) >= REPRESENTATION_RECORD_LIMIT:
            return hidden_by_target
        hidden_by_target.append((model._final_hidden_scalars(context), target))
    for anchor in target_floor_anchors:
        if len(hidden_by_target) >= REPRESENTATION_RECORD_LIMIT:
            break
        context, target, _predicted, _profile = branch_replay_parts(anchor)
        hidden_by_target.append((model._final_hidden_scalars(context), target))
    return hidden_by_target
