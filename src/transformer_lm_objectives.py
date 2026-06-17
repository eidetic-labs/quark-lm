"""Core next-token and branch-diversity objectives for TinyTransformerLM."""

from __future__ import annotations

import math

from autograd import Scalar, zero_grad
from transformer_math import (
    cross_entropy_scalars,
    softmax_floats,
    softmax_scalars,
)


class TransformerObjectiveMixin:
    def predict(self, context: list[int]) -> list[float]:
        return softmax_floats(self._forward_floats(context))

    def nll(self, context: list[int], target: int) -> float:
        probs = self.predict(context)
        return -math.log(max(probs[target], 1e-12))

    def apply_gradients(
        self,
        params: list[Scalar],
        learning_rate: float,
    ) -> float:
        optimizer = self.active_optimizer
        if optimizer is not None:
            return optimizer.apply(params, learning_rate)
        for parameter in params:
            clipped_grad = max(min(parameter.grad, 5.0), -5.0)
            parameter.data -= learning_rate * clipped_grad
        return learning_rate

    def train_step(
        self,
        context: list[int],
        target: int,
        learning_rate: float,
        params: list[Scalar] | None = None,
    ) -> float:
        params = self.parameters() if params is None else params
        zero_grad(params)
        loss = cross_entropy_scalars(self._forward_scalars(context), target)
        loss.backward()
        self.apply_gradients(params, learning_rate)
        return loss.data

    def train_step_with_unlikelihood(
        self,
        context: list[int],
        target: int,
        negative: int,
        learning_rate: float,
        negative_weight: float,
        params: list[Scalar] | None = None,
    ) -> float:
        params = self.parameters() if params is None else params
        zero_grad(params)
        probs = softmax_scalars(self._forward_scalars(context))
        loss = -probs[target].log()
        if negative != target and negative_weight > 0.0:
            loss = loss + (-(Scalar(1.0) - probs[negative] + 1e-12).log()) * negative_weight
        loss.backward()
        self.apply_gradients(params, learning_rate)
        return loss.data

    def train_step_with_unlikelihood_and_positive(
        self,
        context: list[int],
        target: int,
        negative: int,
        positive_context: list[int],
        positive_target: int,
        learning_rate: float,
        negative_weight: float,
        positive_weight: float,
        params: list[Scalar] | None = None,
    ) -> float:
        params = self.parameters() if params is None else params
        zero_grad(params)
        probs = softmax_scalars(self._forward_scalars(context))
        loss = -probs[target].log()
        if negative != target and negative_weight > 0.0:
            loss = loss + (-(Scalar(1.0) - probs[negative] + 1e-12).log()) * negative_weight
        if positive_weight > 0.0:
            positive_probs = softmax_scalars(self._forward_scalars(positive_context))
            loss = loss + (-positive_probs[positive_target].log()) * positive_weight
        loss.backward()
        self.apply_gradients(params, learning_rate)
        return loss.data

    def train_step_with_branch_contrast(
        self,
        context: list[int],
        target: int,
        contrast_context: list[int],
        contrast_target: int,
        learning_rate: float,
        negative_weight: float,
        contrast_weight: float,
        params: list[Scalar] | None = None,
    ) -> float:
        params = self.parameters() if params is None else params
        zero_grad(params)
        probs = softmax_scalars(self._forward_scalars(context))
        loss = -probs[target].log()
        if contrast_target != target and negative_weight > 0.0:
            loss = loss + (-(Scalar(1.0) - probs[contrast_target] + 1e-12).log()) * negative_weight
        if contrast_weight > 0.0:
            contrast_probs = softmax_scalars(self._forward_scalars(contrast_context))
            loss = loss + (-contrast_probs[contrast_target].log()) * contrast_weight
            if target != contrast_target and negative_weight > 0.0:
                loss = loss + (
                    -(Scalar(1.0) - contrast_probs[target] + 1e-12).log()
                ) * negative_weight * contrast_weight
        loss.backward()
        self.apply_gradients(params, learning_rate)
        return loss.data

    def train_step_with_branch_batch_contrast(
        self,
        branches: list[tuple[list[int], int]],
        learning_rate: float,
        negative_weight: float,
        positive_weight: float,
        params: list[Scalar] | None = None,
    ) -> float:
        params = self.parameters() if params is None else params
        zero_grad(params)
        branch_targets = sorted({target for _context, target in branches})
        loss = Scalar(0.0)
        for context, target in branches:
            probs = softmax_scalars(self._forward_scalars(context))
            if positive_weight > 0.0:
                loss = loss + (-probs[target].log()) * positive_weight
            negatives = [
                branch_target
                for branch_target in branch_targets
                if branch_target != target
            ]
            if negative_weight > 0.0 and negatives:
                per_negative_weight = negative_weight / len(negatives)
                for negative in negatives:
                    loss = loss + (
                        -(Scalar(1.0) - probs[negative] + 1e-12).log()
                    ) * per_negative_weight
        loss = loss / max(len(branches), 1)
        loss.backward()
        self.apply_gradients(params, learning_rate)
        return loss.data

    def train_step_with_branch_diversity(
        self,
        branches: list[tuple[list[int], int, int]],
        learning_rate: float,
        negative_weight: float,
        positive_weight: float,
        contrast_weight: float,
        params: list[Scalar] | None = None,
    ) -> float:
        params = self.parameters() if params is None else params
        zero_grad(params)
        branch_targets = sorted({target for _context, target, _predicted in branches})
        loss = Scalar(0.0)
        for context, target, predicted in branches:
            probs = softmax_scalars(self._forward_scalars(context))
            if positive_weight > 0.0:
                loss = loss + (-probs[target].log()) * positive_weight
            if negative_weight > 0.0 and predicted != target:
                loss = loss + (
                    -(Scalar(1.0) - probs[predicted] + 1e-12).log()
                ) * negative_weight
            contrast_targets = [
                branch_target
                for branch_target in branch_targets
                if branch_target != target
            ]
            if contrast_weight > 0.0 and contrast_targets:
                per_target_weight = contrast_weight / len(contrast_targets)
                for contrast_target in contrast_targets:
                    loss = loss + (
                        -(Scalar(1.0) - probs[contrast_target] + 1e-12).log()
                    ) * per_target_weight
        loss = loss / max(len(branches), 1)
        loss.backward()
        self.apply_gradients(params, learning_rate)
        return loss.data
