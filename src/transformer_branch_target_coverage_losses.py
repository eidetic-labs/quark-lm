"""Loss helpers for branch target coverage objectives."""

from __future__ import annotations

from autograd import Scalar


def add_branch_prediction_loss(
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


def target_candidate_ids(
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


def target_set_coverage_loss(
    candidate_probs: list[Scalar],
    candidate_ids: list[int],
    target_set: set[int],
) -> Scalar:
    target_set_mass = target_set_mass_from_candidates(
        candidate_probs,
        candidate_ids,
        target_set,
    )
    return -(target_set_mass + 1e-12).log()


def target_set_mass_from_candidates(
    candidate_probs: list[Scalar],
    candidate_ids: list[int],
    target_set: set[int],
) -> Scalar:
    target_set_mass = Scalar(0.0)
    for offset, candidate_id in enumerate(candidate_ids):
        if candidate_id in target_set:
            target_set_mass = target_set_mass + candidate_probs[offset]
    return target_set_mass


def add_target_shares(
    target_share_sums: list[Scalar],
    candidate_probs: list[Scalar],
    target_set_mass: Scalar,
) -> None:
    for offset, _target in enumerate(target_share_sums):
        target_share_sums[offset] = target_share_sums[offset] + (
            candidate_probs[offset] / (target_set_mass + 1e-12)
        )


def target_balance_loss(
    target_share_sums: list[Scalar],
    branch_count: int,
) -> Scalar:
    target_balance_loss = Scalar(0.0)
    for target_share_sum in target_share_sums:
        average_target_share = target_share_sum / max(branch_count, 1)
        target_balance_loss = target_balance_loss + (
            -(average_target_share + 1e-12).log()
        )
    return target_balance_loss / max(len(target_share_sums), 1)
