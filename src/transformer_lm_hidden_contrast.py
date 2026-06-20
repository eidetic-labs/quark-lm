"""Hidden-state contrast helpers for transformer branch objectives."""

from __future__ import annotations

from autograd import Scalar


def pairwise_hidden_contrast_loss(
    hidden_by_target: list[tuple[list[Scalar], int]],
    embedding_dim: int,
) -> Scalar:
    contrast_loss = Scalar(0.0)
    contrast_pairs = 0
    for left_index, (left_hidden, left_target) in enumerate(hidden_by_target):
        for right_hidden, right_target in hidden_by_target[left_index + 1:]:
            if left_target == right_target:
                continue
            contrast_loss = contrast_loss + _distance_pressure(
                left_hidden,
                right_hidden,
                embedding_dim,
            )
            contrast_pairs += 1
    if not contrast_pairs:
        return Scalar(0.0)
    return contrast_loss / contrast_pairs


def _distance_pressure(
    left_hidden: list[Scalar],
    right_hidden: list[Scalar],
    embedding_dim: int,
) -> Scalar:
    distance_sq = Scalar(0.0)
    for left_value, right_value in zip(left_hidden, right_hidden):
        delta = left_value - right_value
        distance_sq = distance_sq + delta * delta
    distance_sq = distance_sq / max(embedding_dim, 1)
    return (-distance_sq).exp()
