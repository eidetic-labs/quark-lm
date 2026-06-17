from __future__ import annotations

import random

from support.core import (
    ANSWER_TERMINATOR,
    AnswerExample,
    CharTokenizer,
    TinyTransformerLM,
    TransformerConfig,
)
from support.direct_answer import (
    direct_answer_branch_target_ids,
    direct_answer_lesson,
    direct_answer_target_balanced_branch_diversity_batch,
    train_direct_answer_branch_target_diversity_unlikelihood,
    train_direct_answer_branch_target_replay_coverage_unlikelihood,
    train_direct_answer_branch_target_set_coverage_unlikelihood,
)

BranchBatch = list[tuple[list[int], int, int]]


def branch_target_fixture(
    include_blue: bool = False,
) -> tuple[AnswerExample, list[AnswerExample], CharTokenizer]:
    near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
    green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
    tree = AnswerExample(prompt="q: owner?\na:", target=" tree.", source="qa:owner")
    examples = [near, green, tree]
    if include_blue:
        examples.append(
            AnswerExample(prompt="q: thing?\na:", target=" blue.", source="qa:thing")
        )
    text = "".join(example.prompt + example.target for example in examples)
    tokenizer = CharTokenizer.train(text + ANSWER_TERMINATOR)
    return near, examples, tokenizer


def initialized_target_model(
    tokenizer: CharTokenizer,
    *,
    seed: int,
    token_biases: dict[str, float],
) -> TinyTransformerLM:
    model = TinyTransformerLM.init_random(
        TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=8,
            embedding_dim=4,
            feedforward_dim=8,
            seed=seed,
        )
    )
    for token, bias in token_biases.items():
        model.bout[tokenizer.stoi[token]].data = bias
    return model


def target_balanced_branch_batch(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    lesson_example: AnswerExample,
    examples: list[AnswerExample],
    *,
    rng_seed: int,
    batch_size: int,
) -> BranchBatch:
    return direct_answer_target_balanced_branch_diversity_batch(
        model,
        tokenizer,
        lesson_example,
        examples,
        random.Random(rng_seed),
        branch_position=1,
        batch_size=batch_size,
        terminator=ANSWER_TERMINATOR,
    )


def branch_targets_from_batch(batch: BranchBatch) -> list[int]:
    return sorted({target for _context, target, _predicted in batch})


def replay_target_ids(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    examples: list[AnswerExample],
) -> list[int]:
    return direct_answer_branch_target_ids(
        model,
        tokenizer,
        examples,
        branch_position=1,
        terminator=ANSWER_TERMINATOR,
    )


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


def direct_answer_training_lesson(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    lesson_example: AnswerExample,
) -> dict[str, object]:
    return direct_answer_lesson(
        tokenizer,
        model.config.context_size,
        lesson_example,
        ANSWER_TERMINATOR,
    )


def train_target_set_coverage_steps(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    lesson_example: AnswerExample,
    examples: list[AnswerExample],
    *,
    repeat: int,
    rng_seed: int,
) -> None:
    lesson = direct_answer_training_lesson(model, tokenizer, lesson_example)
    rng = random.Random(rng_seed)
    for _ in range(repeat):
        train_direct_answer_branch_target_set_coverage_unlikelihood(
            model,
            tokenizer,
            lesson_example,
            examples,
            lesson,
            rng,
            learning_rate=0.03,
            negative_weight=1.0,
            positive_weight=0.0,
            coverage_weight=2.0,
            branch_position=1,
            batch_size=3,
            hard_negative_count=5,
            terminator=ANSWER_TERMINATOR,
            balance_targets=True,
        )


def train_target_diversity_steps(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    lesson_example: AnswerExample,
    examples: list[AnswerExample],
    *,
    repeat: int,
    rng_seed: int,
) -> None:
    lesson = direct_answer_training_lesson(model, tokenizer, lesson_example)
    rng = random.Random(rng_seed)
    for _ in range(repeat):
        train_direct_answer_branch_target_diversity_unlikelihood(
            model,
            tokenizer,
            lesson_example,
            examples,
            lesson,
            rng,
            learning_rate=0.03,
            negative_weight=1.0,
            positive_weight=0.0,
            diversity_weight=2.0,
            branch_position=1,
            batch_size=3,
            hard_negative_count=5,
            terminator=ANSWER_TERMINATOR,
            balance_targets=True,
        )


def train_target_replay_coverage_steps(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    lesson_example: AnswerExample,
    examples: list[AnswerExample],
    *,
    repeat: int,
    rng_seed: int,
) -> None:
    lesson = direct_answer_training_lesson(model, tokenizer, lesson_example)
    rng = random.Random(rng_seed)
    for _ in range(repeat):
        train_direct_answer_branch_target_replay_coverage_unlikelihood(
            model,
            tokenizer,
            lesson_example,
            examples,
            lesson,
            rng,
            learning_rate=0.03,
            negative_weight=1.0,
            positive_weight=0.0,
            replay_weight=2.0,
            branch_position=1,
            batch_size=2,
            hard_negative_count=5,
            terminator=ANSWER_TERMINATOR,
            balance_targets=True,
        )
