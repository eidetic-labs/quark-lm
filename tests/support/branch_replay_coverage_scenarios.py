from __future__ import annotations

from dataclasses import dataclass

from support.branch_replay_coverage_fixtures import (
    BranchBatch,
    target_balanced_branch_batch,
)
from support.core import AnswerExample, CharTokenizer, TinyTransformerLM


@dataclass(frozen=True)
class ReplayDeficitScenario:
    replay_branches: BranchBatch
    replay_targets: list[int]
    deficit_targets: set[int]
    deficit_context: list[int]
    deficit_target: int
    represented_context: list[int]
    represented_prediction: int


def replay_deficit_scenario(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    lesson_example: AnswerExample,
    examples: list[AnswerExample],
) -> ReplayDeficitScenario:
    replay_branches = target_balanced_branch_batch(
        model,
        tokenizer,
        lesson_example,
        examples,
        rng_seed=16,
        batch_size=3,
    )
    replay_targets = sorted(
        {target for _context, target, _predicted in replay_branches}
    )
    replay_target_set = set(replay_targets)
    predicted_replay_targets = {
        predicted
        for _context, _target, predicted in replay_branches
        if predicted in replay_target_set
    }
    deficit_targets = replay_target_set - predicted_replay_targets
    if not deficit_targets:
        raise AssertionError("Expected at least one uncovered replay target")

    deficit_context, deficit_target, _predicted = next(
        branch for branch in replay_branches if branch[1] in deficit_targets
    )
    represented_context, _represented_target, represented_prediction = next(
        branch for branch in replay_branches if branch[2] in predicted_replay_targets
    )
    return ReplayDeficitScenario(
        replay_branches=replay_branches,
        replay_targets=replay_targets,
        deficit_targets=deficit_targets,
        deficit_context=deficit_context,
        deficit_target=deficit_target,
        represented_context=represented_context,
        represented_prediction=represented_prediction,
    )


def target_probability(
    model: TinyTransformerLM,
    scenario: ReplayDeficitScenario,
    context: list[int],
    target: int,
) -> float:
    probs = model.predict(context)
    replay_target_set = set(scenario.replay_targets)
    hard_candidates = [
        index
        for index in sorted(
            range(len(probs)),
            key=lambda item: probs[item],
            reverse=True,
        )
        if index not in replay_target_set
    ][:5]
    candidate_ids = [*scenario.replay_targets, *hard_candidates]
    denominator = sum(probs[candidate_id] for candidate_id in candidate_ids)
    return probs[target] / denominator
