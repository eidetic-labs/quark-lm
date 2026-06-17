"""Balanced target-loss helpers for branch-context objectives."""

from __future__ import annotations

from collections import Counter

from autograd import Scalar


def balanced_target_loss(
    losses_by_target: dict[tuple[str, int], Scalar],
    counts_by_target: Counter[tuple[str, int]],
) -> Scalar:
    balanced_loss = Scalar(0.0)
    for target, target_loss in losses_by_target.items():
        balanced_loss = balanced_loss + (target_loss / counts_by_target[target])
    return balanced_loss / len(losses_by_target)
