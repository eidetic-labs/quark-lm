"""Bidirectional and coverage branch-binding objectives."""

from __future__ import annotations

from autograd import Scalar, zero_grad
from transformer_math import softmax_scalars


class TransformerBranchTargetBindingMixin:
    def train_step_with_branch_bidirectional_binding(
        self,
        branches: list[tuple[list[int], int, int]],
        learning_rate: float,
        negative_weight: float,
        positive_weight: float,
        binding_weight: float,
        params: list[Scalar] | None = None,
    ) -> float:
        params = self.parameters() if params is None else params
        zero_grad(params)
        branch_targets = sorted({target for _context, target, _predicted in branches})
        branch_target_offsets = {
            target: offset for offset, target in enumerate(branch_targets)
        }
        branch_loss = Scalar(0.0)
        branch_logits_by_target: list[tuple[list[Scalar], int]] = []
        for context, target, predicted in branches:
            logits = self._forward_scalars(context)
            probs = softmax_scalars(logits)
            branch_loss = _add_branch_prediction_loss(
                branch_loss,
                probs,
                target,
                predicted,
                negative_weight,
                positive_weight,
            )
            branch_logits_by_target.append((logits, target))
        loss = branch_loss / max(len(branches), 1)
        if binding_weight > 0.0 and len(branch_targets) > 1:
            binding_loss = _bidirectional_target_binding_loss(
                branch_logits_by_target,
                branch_targets,
                branch_target_offsets,
            )
            loss = loss + binding_loss * binding_weight
        loss.backward()
        self.apply_gradients(params, learning_rate)
        return loss.data

    def train_step_with_branch_coverage_binding(
        self,
        branches: list[tuple[list[int], int, int]],
        learning_rate: float,
        negative_weight: float,
        positive_weight: float,
        binding_weight: float,
        hard_negative_count: int,
        params: list[Scalar] | None = None,
    ) -> float:
        params = self.parameters() if params is None else params
        zero_grad(params)
        branch_targets = sorted({target for _context, target, _predicted in branches})
        branch_target_set = set(branch_targets)
        branch_loss = Scalar(0.0)
        row_loss = Scalar(0.0)
        coverage_loss = Scalar(0.0)
        branch_logits_by_target: list[tuple[list[Scalar], int]] = []
        for context, target, predicted in branches:
            logits = self._forward_scalars(context)
            probs = softmax_scalars(logits)
            branch_loss = _add_branch_prediction_loss(
                branch_loss,
                probs,
                target,
                predicted,
                negative_weight,
                positive_weight,
            )
            if binding_weight > 0.0 and len(branch_targets) > 1:
                candidate_ids = _target_candidate_ids(
                    logits,
                    branch_targets,
                    branch_target_set,
                    self.config.vocab_size,
                    hard_negative_count,
                )
                candidate_probs = softmax_scalars(
                    [logits[candidate_id] for candidate_id in candidate_ids]
                )
                target_offset = candidate_ids.index(target)
                row_loss = row_loss + (-candidate_probs[target_offset].log())
                coverage_loss = coverage_loss + _target_set_coverage_loss(
                    candidate_probs,
                    candidate_ids,
                    branch_target_set,
                )
            branch_logits_by_target.append((logits, target))
        loss = branch_loss / max(len(branches), 1)
        if binding_weight > 0.0 and len(branch_targets) > 1:
            binding_loss = _coverage_binding_loss(
                row_loss,
                coverage_loss,
                branch_logits_by_target,
                branch_targets,
            )
            loss = loss + binding_loss * binding_weight
        loss.backward()
        self.apply_gradients(params, learning_rate)
        return loss.data


def _add_branch_prediction_loss(
    branch_loss: Scalar,
    probs: list[Scalar],
    target: int,
    predicted: int,
    negative_weight: float,
    positive_weight: float,
) -> Scalar:
    if positive_weight > 0.0:
        branch_loss = branch_loss + (-probs[target].log()) * positive_weight
    if negative_weight > 0.0 and predicted != target:
        branch_loss = branch_loss + (
            -(Scalar(1.0) - probs[predicted] + 1e-12).log()
        ) * negative_weight
    return branch_loss


def _bidirectional_target_binding_loss(
    branch_logits_by_target: list[tuple[list[Scalar], int]],
    branch_targets: list[int],
    branch_target_offsets: dict[int, int],
) -> Scalar:
    row_loss = Scalar(0.0)
    for logits, target in branch_logits_by_target:
        target_logits = [logits[branch_target] for branch_target in branch_targets]
        target_probs = softmax_scalars(target_logits)
        row_loss = row_loss + (-target_probs[branch_target_offsets[target]].log())
    row_loss = row_loss / max(len(branch_logits_by_target), 1)
    column_loss, column_count = _column_binding_loss(
        branch_logits_by_target,
        branch_targets,
    )
    if column_count:
        return (row_loss + column_loss / column_count) / 2.0
    return row_loss


def _coverage_binding_loss(
    row_loss: Scalar,
    coverage_loss: Scalar,
    branch_logits_by_target: list[tuple[list[Scalar], int]],
    branch_targets: list[int],
) -> Scalar:
    row_loss = row_loss / max(len(branch_logits_by_target), 1)
    coverage_loss = coverage_loss / max(len(branch_logits_by_target), 1)
    binding_loss = (row_loss + coverage_loss) / 2.0
    column_loss, column_count = _column_binding_loss(
        branch_logits_by_target,
        branch_targets,
    )
    if column_count:
        binding_loss = (binding_loss + column_loss / column_count) / 2.0
    return binding_loss


def _column_binding_loss(
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


def _target_candidate_ids(
    logits: list[Scalar],
    targets: list[int],
    target_set: set[int],
    vocab_size: int,
    hard_negative_count: int,
) -> list[int]:
    hard_candidates = [
        index
        for index in sorted(
            range(vocab_size),
            key=lambda item: logits[item].data,
            reverse=True,
        )
        if index not in target_set
    ]
    if hard_negative_count > 0:
        hard_candidates = hard_candidates[:hard_negative_count]
    return [*targets, *hard_candidates]


def _target_set_coverage_loss(
    candidate_probs: list[Scalar],
    candidate_ids: list[int],
    target_set: set[int],
) -> Scalar:
    target_set_mass = Scalar(0.0)
    for offset, candidate_id in enumerate(candidate_ids):
        if candidate_id in target_set:
            target_set_mass = target_set_mass + candidate_probs[offset]
    return -(target_set_mass + 1e-12).log()
