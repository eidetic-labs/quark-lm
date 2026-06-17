from __future__ import annotations

from support.branch_target_coverage_fixtures import BranchBatch
from support.core import TinyTransformerLM


def hard_candidate_ids(
    model: TinyTransformerLM,
    context: list[int],
    excluded_targets: set[int],
) -> list[int]:
    probs = model.predict(context)
    return [
        index
        for index in sorted(
            range(len(probs)),
            key=lambda item: probs[item],
            reverse=True,
        )
        if index not in excluded_targets
    ][:5]


def target_values(
    model: TinyTransformerLM,
    context: list[int],
    targets: list[int],
) -> list[float]:
    probs = model.predict(context)
    candidate_ids = [*targets, *hard_candidate_ids(model, context, set(targets))]
    denominator = sum(probs[candidate_id] for candidate_id in candidate_ids)
    return [probs[target] / denominator for target in targets]


def restricted_target_set_mass(
    model: TinyTransformerLM,
    batch: BranchBatch,
    branch_targets: list[int],
) -> float:
    total = 0.0
    for context, _target, _predicted in batch:
        total += sum(target_values(model, context, branch_targets))
    return total / len(batch)


def restricted_target_metrics(
    model: TinyTransformerLM,
    batch: BranchBatch,
    branch_targets: list[int],
) -> tuple[float, float]:
    target_set_total = 0.0
    target_share_totals = [0.0 for _branch_target in branch_targets]
    for context, _target, _predicted in batch:
        values = target_values(model, context, branch_targets)
        target_set_mass = sum(values)
        target_set_total += target_set_mass
        for offset, value in enumerate(values):
            target_share_totals[offset] += value / target_set_mass
    average_target_shares = [
        target_share_total / len(batch) for target_share_total in target_share_totals
    ]
    return target_set_total / len(batch), min(average_target_shares)


def replay_target_metrics(
    model: TinyTransformerLM,
    batch: BranchBatch,
    replay_targets: list[int],
    missing_targets: set[int],
) -> tuple[float, float]:
    target_set_total = 0.0
    missing_share_totals = [0.0 for _missing_target in missing_targets]
    missing_offsets = [
        offset
        for offset, replay_target in enumerate(replay_targets)
        if replay_target in missing_targets
    ]
    for context, _target, _predicted in batch:
        values = target_values(model, context, replay_targets)
        target_set_mass = sum(values)
        target_set_total += target_set_mass
        for missing_index, target_offset in enumerate(missing_offsets):
            missing_share_totals[missing_index] += (
                values[target_offset] / target_set_mass
            )
    average_missing_shares = [
        missing_share_total / len(batch)
        for missing_share_total in missing_share_totals
    ]
    return target_set_total / len(batch), min(average_missing_shares)
