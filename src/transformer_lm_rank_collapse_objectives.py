"""Rank-margin plus anti-collapse branch objectives for TinyTransformerLM."""

from __future__ import annotations

from collections import Counter

from autograd import Scalar, zero_grad
from transformer_lm_hidden_contrast import pairwise_hidden_contrast_loss
from transformer_math import linear_scalars, softmax_scalars


class TransformerRankCollapseObjectiveMixin:
    def train_step_with_branch_rank_collapse_margin(
        self,
        branches: list[tuple[list[int], int, int]],
        learning_rate: float,
        negative_weight: float,
        positive_weight: float,
        margin_weight: float,
        collapse_weight: float,
        hard_negative_count: int,
        params: list[Scalar] | None = None,
    ) -> float:
        params = self.parameters() if params is None else params
        zero_grad(params)
        dominant_wrong = _dominant_wrong_prediction(branches)
        output_weights = self._output_weights_scalars()
        hidden_by_target: list[tuple[list[Scalar], int]] = []
        loss = Scalar(0.0)
        for context, target, predicted in branches:
            hidden = self._final_hidden_scalars(context)
            logits = linear_scalars(hidden, output_weights, self.bout)
            probs = softmax_scalars(logits)
            hidden_by_target.append((hidden, target))
            if positive_weight > 0.0:
                loss = loss + (-probs[target].log()) * positive_weight
            if negative_weight > 0.0 and predicted != target:
                loss = loss + (
                    -(Scalar(1.0) - probs[predicted] + 1e-12).log()
                ) * negative_weight
            if (
                collapse_weight > 0.0
                and dominant_wrong is not None
                and dominant_wrong != target
            ):
                loss = loss + (
                    -(Scalar(1.0) - probs[dominant_wrong] + 1e-12).log()
                ) * collapse_weight
            hard_negatives = _hard_negative_ids(
                logits,
                target,
                hard_negative_count,
            )
            if margin_weight > 0.0 and hard_negatives:
                target_logit = logits[target]
                per_negative_weight = margin_weight / len(hard_negatives)
                for hard_negative in hard_negatives:
                    gap = logits[hard_negative] - target_logit + 1.0
                    loss = loss + (
                        (Scalar(1.0) + gap.exp()).log() * per_negative_weight
                    )
        loss = loss / max(len(branches), 1)
        if collapse_weight > 0.0:
            loss = loss + (
                pairwise_hidden_contrast_loss(
                    hidden_by_target,
                    self.config.embedding_dim,
                )
                * collapse_weight
            )
        loss.backward()
        self.apply_gradients(params, learning_rate)
        return loss.data


def _dominant_wrong_prediction(
    branches: list[tuple[list[int], int, int]],
) -> int | None:
    counts: Counter[int] = Counter()
    for _context, target, predicted in branches:
        if predicted != target:
            counts[predicted] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _hard_negative_ids(
    logits: list[Scalar],
    target: int,
    hard_negative_count: int,
) -> list[int]:
    hard_negatives = [
        index
        for index in sorted(
            range(len(logits)),
            key=lambda item: logits[item].data,
            reverse=True,
        )
        if index != target
    ]
    if hard_negative_count > 0:
        return hard_negatives[:hard_negative_count]
    return hard_negatives
