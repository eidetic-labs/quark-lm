"""Branch-target objective variants for TinyTransformerLM."""

from __future__ import annotations

from autograd import Scalar, zero_grad
from transformer_math import linear_scalars, softmax_scalars


def branch_target_candidate_ids(
    branch_targets: list[int],
    predicted: int,
) -> list[int]:
    return sorted({*branch_targets, predicted})


class TransformerBranchTargetObjectiveMixin:
    def train_step_with_branch_target_softmax(
        self,
        branches: list[tuple[list[int], int, int]],
        learning_rate: float,
        negative_weight: float,
        positive_weight: float,
        target_softmax_weight: float,
        params: list[Scalar] | None = None,
    ) -> float:
        params = self.parameters() if params is None else params
        zero_grad(params)
        branch_targets = sorted({target for _context, target, _predicted in branches})
        loss = Scalar(0.0)
        for context, target, predicted in branches:
            logits = self._forward_scalars(context)
            probs = softmax_scalars(logits)
            if positive_weight > 0.0:
                loss = loss + (-probs[target].log()) * positive_weight
            if negative_weight > 0.0 and predicted != target:
                loss = loss + (
                    -(Scalar(1.0) - probs[predicted] + 1e-12).log()
                ) * negative_weight
            target_candidates = branch_target_candidate_ids(branch_targets, predicted)
            if target_softmax_weight > 0.0 and len(target_candidates) > 1:
                target_offsets = {
                    candidate: offset
                    for offset, candidate in enumerate(target_candidates)
                }
                target_logits = [
                    logits[candidate] for candidate in target_candidates
                ]
                target_probs = softmax_scalars(target_logits)
                loss = loss + (
                    -target_probs[target_offsets[target]].log()
                ) * target_softmax_weight
        loss = loss / max(len(branches), 1)
        loss.backward()
        self.apply_gradients(params, learning_rate)
        return loss.data

    def train_step_with_branch_target_margin(
        self,
        branches: list[tuple[list[int], int, int]],
        learning_rate: float,
        negative_weight: float,
        positive_weight: float,
        margin_weight: float,
        params: list[Scalar] | None = None,
    ) -> float:
        params = self.parameters() if params is None else params
        zero_grad(params)
        branch_targets = sorted({target for _context, target, _predicted in branches})
        loss = Scalar(0.0)
        for context, target, predicted in branches:
            logits = self._forward_scalars(context)
            probs = softmax_scalars(logits)
            if positive_weight > 0.0:
                loss = loss + (-probs[target].log()) * positive_weight
            if negative_weight > 0.0 and predicted != target:
                loss = loss + (
                    -(Scalar(1.0) - probs[predicted] + 1e-12).log()
                ) * negative_weight
            margin_targets = [
                candidate
                for candidate in branch_target_candidate_ids(branch_targets, predicted)
                if candidate != target
            ]
            if margin_weight > 0.0 and margin_targets:
                per_target_weight = margin_weight / len(margin_targets)
                target_logit = logits[target]
                for margin_target in margin_targets:
                    gap = logits[margin_target] - target_logit + 1.0
                    loss = loss + (Scalar(1.0) + gap.exp()).log() * per_target_weight
        loss = loss / max(len(branches), 1)
        loss.backward()
        self.apply_gradients(params, learning_rate)
        return loss.data

    def train_step_with_branch_hidden_projection_margin(
        self,
        branches: list[tuple[list[int], int, int]],
        learning_rate: float,
        negative_weight: float,
        positive_weight: float,
        margin_weight: float,
        params: list[Scalar] | None = None,
    ) -> float:
        params = self.parameters() if params is None else params
        zero_grad(params)
        branch_targets = sorted({target for _context, target, _predicted in branches})
        output_weights = self._output_weights_scalars()
        loss = Scalar(0.0)
        for context, target, predicted in branches:
            hidden = self._final_hidden_scalars(context)
            logits = linear_scalars(hidden, output_weights, self.bout)
            probs = softmax_scalars(logits)
            if positive_weight > 0.0:
                loss = loss + (-probs[target].log()) * positive_weight
            if negative_weight > 0.0 and predicted != target:
                loss = loss + (
                    -(Scalar(1.0) - probs[predicted] + 1e-12).log()
                ) * negative_weight
            margin_targets = [
                branch_target
                for branch_target in branch_targets
                if branch_target != target
            ]
            if margin_weight > 0.0 and margin_targets:
                per_target_weight = margin_weight / len(margin_targets)
                target_projection = Scalar(0.0)
                for dim, value in enumerate(hidden):
                    target_projection = (
                        target_projection + value * output_weights[dim][target]
                    )
                for margin_target in margin_targets:
                    margin_projection = Scalar(0.0)
                    for dim, value in enumerate(hidden):
                        margin_projection = (
                            margin_projection
                            + value * output_weights[dim][margin_target]
                        )
                    gap = margin_projection - target_projection + 1.0
                    loss = loss + (Scalar(1.0) + gap.exp()).log() * per_target_weight
        loss = loss / max(len(branches), 1)
        loss.backward()
        self.apply_gradients(params, learning_rate)
        return loss.data
