from __future__ import annotations

import math

from support.branch_binding_fixtures import BranchBatch
from support.core import TinyTransformerLM


def average_target_rank(model: TinyTransformerLM, batch: BranchBatch) -> float:
    total = 0.0
    for context, target, _predicted in batch:
        probs = model.predict(context)
        ranked = sorted(
            range(len(probs)),
            key=lambda index: probs[index],
            reverse=True,
        )
        total += ranked.index(target) + 1
    return total / len(batch)


def average_target_context_ownership(
    model: TinyTransformerLM,
    batch: BranchBatch,
    branch_targets: list[int],
) -> float:
    total = 0.0
    for branch_target in branch_targets:
        target_logits = [
            model._forward_floats(context)[branch_target]
            for context, _target, _predicted in batch
        ]
        max_logit = max(target_logits)
        exp_scores = [
            math.exp(target_logit - max_logit) for target_logit in target_logits
        ]
        denominator = sum(exp_scores)
        owned_mass = 0.0
        for exp_score, (_context, target, _predicted) in zip(exp_scores, batch):
            if target == branch_target:
                owned_mass += exp_score / denominator
        total += owned_mass
    return total / len(branch_targets)


def restricted_probabilities(
    model: TinyTransformerLM,
    batch: BranchBatch,
    branch_targets: list[int],
) -> tuple[float, float]:
    target_set_total = 0.0
    target_total = 0.0
    branch_target_set = set(branch_targets)
    for context, target, _predicted in batch:
        probs = model.predict(context)
        hard_candidates = [
            index
            for index in sorted(
                range(len(probs)),
                key=lambda item: probs[item],
                reverse=True,
            )
            if index not in branch_target_set
        ][:5]
        candidate_ids = [*branch_targets, *hard_candidates]
        denominator = sum(probs[candidate_id] for candidate_id in candidate_ids)
        target_set_total += (
            sum(probs[branch_target] for branch_target in branch_targets)
            / denominator
        )
        target_total += probs[target] / denominator
    return target_set_total / len(batch), target_total / len(batch)
