"""Rank and top-k branch objectives for TinyTransformerLM."""

from __future__ import annotations

from autograd import Scalar, zero_grad
from transformer_math import softmax_scalars


class TransformerRankObjectiveMixin:
    def train_step_with_branch_rank_margin(
        self,
        branches: list[tuple[list[int], int, int]],
        learning_rate: float,
        negative_weight: float,
        positive_weight: float,
        margin_weight: float,
        hard_negative_count: int,
        params: list[Scalar] | None = None,
    ) -> float:
        params = self.parameters() if params is None else params
        zero_grad(params)
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
            hard_negatives = [
                index
                for index in sorted(
                    range(self.config.vocab_size),
                    key=lambda item: logits[item].data,
                    reverse=True,
                )
                if index != target
            ]
            if hard_negative_count > 0:
                hard_negatives = hard_negatives[:hard_negative_count]
            if margin_weight > 0.0 and hard_negatives:
                per_negative_weight = margin_weight / len(hard_negatives)
                target_logit = logits[target]
                for hard_negative in hard_negatives:
                    gap = logits[hard_negative] - target_logit + 1.0
                    loss = loss + (
                        (Scalar(1.0) + gap.exp()).log() * per_negative_weight
                    )
        loss = loss / max(len(branches), 1)
        loss.backward()
        self.apply_gradients(params, learning_rate)
        return loss.data

    def train_step_with_branch_topk_softmax(
        self,
        branches: list[tuple[list[int], int, int]],
        learning_rate: float,
        negative_weight: float,
        positive_weight: float,
        candidate_weight: float,
        candidate_count: int,
        params: list[Scalar] | None = None,
    ) -> float:
        params = self.parameters() if params is None else params
        zero_grad(params)
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
            hard_candidates = [
                index
                for index in sorted(
                    range(self.config.vocab_size),
                    key=lambda item: logits[item].data,
                    reverse=True,
                )
                if index != target
            ]
            if candidate_count > 0:
                hard_candidates = hard_candidates[:candidate_count]
            candidate_ids = [target, *hard_candidates]
            if candidate_weight > 0.0 and len(candidate_ids) > 1:
                candidate_logits = [
                    logits[candidate_id] for candidate_id in candidate_ids
                ]
                candidate_probs = softmax_scalars(candidate_logits)
                loss = loss + (-candidate_probs[0].log()) * candidate_weight
        loss = loss / max(len(branches), 1)
        loss.backward()
        self.apply_gradients(params, learning_rate)
        return loss.data
