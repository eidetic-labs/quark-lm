"""Candidate-set target losses for branch repair objectives."""

from __future__ import annotations

from autograd import Scalar
from transformer_math import softmax_scalars


def candidate_set_target_loss(
    logits: list[Scalar],
    target: int,
    candidate_weight: float,
    candidate_count: int,
    vocab_size: int,
) -> Scalar:
    """Return cross-entropy for target versus its hardest current candidates."""

    if candidate_weight <= 0.0:
        return Scalar(0.0)
    candidate_ids = _candidate_ids(logits, target, candidate_count, vocab_size)
    if len(candidate_ids) <= 1:
        return Scalar(0.0)
    candidate_logits = [logits[candidate_id] for candidate_id in candidate_ids]
    candidate_probs = softmax_scalars(candidate_logits)
    return (-candidate_probs[0].log()) * candidate_weight


def _candidate_ids(
    logits: list[Scalar],
    target: int,
    candidate_count: int,
    vocab_size: int,
) -> list[int]:
    hard_candidates = [
        index
        for index in sorted(
            range(vocab_size),
            key=lambda item: logits[item].data,
            reverse=True,
        )
        if index != target
    ]
    if candidate_count > 0:
        hard_candidates = hard_candidates[:candidate_count]
    return [target, *hard_candidates]
