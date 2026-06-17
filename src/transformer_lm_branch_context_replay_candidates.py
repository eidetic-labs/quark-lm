"""Candidate sets for branch-context replay losses."""

from __future__ import annotations

from autograd import Scalar


def branch_context_candidate_ids(
    logits: list[Scalar],
    vocab_size: int,
    profile_targets: list[int],
    profile_target_set: set[int],
    hard_negative_count: int,
) -> list[int]:
    hard_candidates = [
        index
        for index in sorted(
            range(vocab_size),
            key=lambda item: logits[item].data,
            reverse=True,
        )
        if index not in profile_target_set
    ]
    if hard_negative_count > 0:
        hard_candidates = hard_candidates[:hard_negative_count]
    return [*profile_targets, *hard_candidates]


def branch_context_target_set_mass(
    candidate_ids: list[int],
    candidate_probs: list[Scalar],
    profile_target_set: set[int],
) -> Scalar:
    target_set_mass = Scalar(0.0)
    for offset, candidate_id in enumerate(candidate_ids):
        if candidate_id in profile_target_set:
            target_set_mass = target_set_mass + candidate_probs[offset]
    return target_set_mass
