"""Target-set binding helpers for transformer branch objectives."""

from __future__ import annotations

from autograd import Scalar
from transformer_math import softmax_scalars


def target_set_row_binding_loss(
    branch_logits_by_target: list[tuple[list[Scalar], int]],
    branch_targets: list[int],
) -> Scalar:
    if len(branch_targets) <= 1:
        return Scalar(0.0)
    target_offsets = {
        target: offset for offset, target in enumerate(branch_targets)
    }
    row_loss = Scalar(0.0)
    for logits, target in branch_logits_by_target:
        target_logits = [logits[branch_target] for branch_target in branch_targets]
        target_probs = softmax_scalars(target_logits)
        row_loss = row_loss + (-target_probs[target_offsets[target]].log())
    return row_loss / max(len(branch_logits_by_target), 1)


def bidirectional_target_binding_loss(
    branch_logits_by_target: list[tuple[list[Scalar], int]],
    branch_targets: list[int],
) -> Scalar:
    row_loss = target_set_row_binding_loss(branch_logits_by_target, branch_targets)
    column_loss, column_count = target_context_column_binding_loss(
        branch_logits_by_target,
        branch_targets,
    )
    if column_count:
        return (row_loss + column_loss / column_count) / 2.0
    return row_loss


def target_context_column_binding_loss(
    branch_logits_by_target: list[tuple[list[Scalar], int]],
    branch_targets: list[int],
) -> tuple[Scalar, int]:
    column_loss = Scalar(0.0)
    column_count = 0
    for branch_target in branch_targets:
        context_logits = [
            logits[branch_target] for logits, _target in branch_logits_by_target
        ]
        positive_indexes = [
            index
            for index, (_logits, target) in enumerate(branch_logits_by_target)
            if target == branch_target
        ]
        if not positive_indexes or len(positive_indexes) == len(context_logits):
            continue
        context_probs = softmax_scalars(context_logits)
        positive_mass = Scalar(0.0)
        for index in positive_indexes:
            positive_mass = positive_mass + context_probs[index]
        column_loss = column_loss + (-(positive_mass + 1e-12).log())
        column_count += 1
    return column_loss, column_count
