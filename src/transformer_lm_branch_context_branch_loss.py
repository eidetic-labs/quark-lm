"""Branch example losses for branch-context replay objectives."""

from __future__ import annotations

from typing import Any

from autograd import Scalar
from transformer_lm_branch_context_loss_state import BranchContextLossState
from transformer_lm_branch_context_targets import BranchReplayPart
from transformer_math import softmax_scalars


def accumulate_branch_losses(
    model: Any,
    branch_parts: list[BranchReplayPart],
    losses: BranchContextLossState,
    *,
    positive_weight: float,
    negative_weight: float,
) -> None:
    for context, target, predicted, _profile in branch_parts:
        logits = model._forward_scalars(context)
        probs = softmax_scalars(logits)
        if positive_weight > 0.0:
            losses.branch_loss = losses.branch_loss + (-probs[target].log()) * (
                positive_weight
            )
        if negative_weight > 0.0 and predicted != target:
            losses.branch_loss = losses.branch_loss + (
                -(Scalar(1.0) - probs[predicted] + 1e-12).log()
            ) * negative_weight
