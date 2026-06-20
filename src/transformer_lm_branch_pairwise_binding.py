"""Pairwise branch representation and output-binding objectives."""

from __future__ import annotations

from autograd import Scalar, zero_grad
from transformer_lm_hidden_contrast import pairwise_hidden_contrast_loss
from transformer_math import linear_scalars, softmax_scalars


class TransformerBranchPairwiseBindingMixin:
    def train_step_with_branch_representation_contrast(
        self,
        branches: list[tuple[list[int], int, int]],
        learning_rate: float,
        negative_weight: float,
        positive_weight: float,
        representation_weight: float,
        params: list[Scalar] | None = None,
    ) -> float:
        params = self.parameters() if params is None else params
        zero_grad(params)
        branch_loss = Scalar(0.0)
        hidden_by_target: list[tuple[list[Scalar], int]] = []
        for context, target, predicted in branches:
            hidden = self._final_hidden_scalars(context)
            logits = linear_scalars(hidden, self.wout, self.bout)
            probs = softmax_scalars(logits)
            branch_loss = _add_branch_prediction_loss(
                branch_loss,
                probs,
                target,
                predicted,
                negative_weight,
                positive_weight,
            )
            hidden_by_target.append((hidden, target))
        loss = branch_loss / max(len(branches), 1)
        if representation_weight > 0.0:
            loss = loss + (
                pairwise_hidden_contrast_loss(
                    hidden_by_target,
                    self.config.embedding_dim,
                )
                * representation_weight
            )
        loss.backward()
        self.apply_gradients(params, learning_rate)
        return loss.data

    def train_step_with_branch_output_binding(
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
        hidden_by_target: list[tuple[list[Scalar], int]] = []
        for context, target, predicted in branches:
            hidden = self._final_hidden_scalars(context)
            logits = linear_scalars(hidden, self.wout, self.bout)
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
                target_logits = [logits[branch_target] for branch_target in branch_targets]
                target_probs = softmax_scalars(target_logits)
                branch_loss = branch_loss + (
                    -target_probs[branch_target_offsets[target]].log()
                ) * binding_weight
            hidden_by_target.append((hidden, target))
        loss = branch_loss / max(len(branches), 1)
        if binding_weight > 0.0:
            loss = loss + (
                pairwise_hidden_contrast_loss(
                    hidden_by_target,
                    self.config.embedding_dim,
                )
                * binding_weight
            )
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
