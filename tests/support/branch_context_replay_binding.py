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
    direct_answer_branch_diversity_batch,
    direct_answer_lesson,
    direct_answer_target_balanced_branch_diversity_batch,
    train_direct_answer_branch_context_replay_coverage_unlikelihood,
)

BranchBatch = list[tuple[list[int], int, int]]


def branch_binding_fixture(
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


def initialized_replay_model(
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


def branch_diversity_batch(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    lesson_example: AnswerExample,
    examples: list[AnswerExample],
    *,
    rng_seed: int,
    batch_size: int,
) -> BranchBatch:
    return direct_answer_branch_diversity_batch(
        model,
        tokenizer,
        lesson_example,
        examples,
        random.Random(rng_seed),
        branch_position=1,
        batch_size=batch_size,
        terminator=ANSWER_TERMINATOR,
    )


def replay_target_ids(replay_branches: BranchBatch) -> list[int]:
    return sorted({target for _context, target, _predicted in replay_branches})


def covered_replay_targets(replay_branches: BranchBatch) -> set[int]:
    return {
        target for _context, target, predicted in replay_branches if target == predicted
    }


def target_normalized_probability(
    model: TinyTransformerLM,
    context: list[int],
    target: int,
    replay_targets: list[int],
) -> float:
    probs = model.predict(context)
    replay_target_set = set(replay_targets)
    hard_candidates = [
        index
        for index in sorted(
            range(len(probs)),
            key=lambda item: probs[item],
            reverse=True,
        )
        if index not in replay_target_set
    ][:5]
    candidate_ids = [*replay_targets, *hard_candidates]
    denominator = sum(probs[candidate_id] for candidate_id in candidate_ids)
    return probs[target] / denominator


def train_anchor_comparison_steps(
    unanchored_model: TinyTransformerLM,
    global_anchor_model: TinyTransformerLM,
    balanced_anchor_model: TinyTransformerLM,
    branch_batch: BranchBatch,
    replay_branches: BranchBatch,
    *,
    repeat: int,
) -> None:
    for _ in range(repeat):
        unanchored_model.train_step_with_branch_context_replay_coverage(
            branch_batch,
            replay_branches,
            learning_rate=0.03,
            negative_weight=1.0,
            positive_weight=0.0,
            replay_weight=2.0,
            hard_negative_count=5,
        )
        global_anchor_model.train_step_with_branch_context_replay_coverage(
            branch_batch,
            replay_branches,
            learning_rate=0.03,
            negative_weight=1.0,
            positive_weight=0.0,
            replay_weight=2.0,
            hard_negative_count=5,
            preserve_covered_targets=True,
        )
        balanced_anchor_model.train_step_with_branch_context_replay_coverage(
            branch_batch,
            replay_branches,
            learning_rate=0.03,
            negative_weight=1.0,
            positive_weight=0.0,
            replay_weight=2.0,
            hard_negative_count=5,
            preserve_covered_targets=True,
            balance_covered_target_anchors=True,
        )


def replay_context_metrics(
    model: TinyTransformerLM,
    replay_branches: BranchBatch,
    replay_targets: list[int],
) -> tuple[float, float]:
    target_set_total = 0.0
    owned_shares = []
    for context, target, _predicted in replay_branches:
        probs = model.predict(context)
        replay_target_set = set(replay_targets)
        hard_candidates = [
            index
            for index in sorted(
                range(len(probs)),
                key=lambda item: probs[item],
                reverse=True,
            )
            if index not in replay_target_set
        ][:5]
        candidate_ids = [*replay_targets, *hard_candidates]
        denominator = sum(probs[candidate_id] for candidate_id in candidate_ids)
        target_values = [
            probs[replay_target] / denominator for replay_target in replay_targets
        ]
        target_set_mass = sum(target_values)
        target_set_total += target_set_mass
        target_offset = replay_targets.index(target)
        owned_shares.append(target_values[target_offset] / target_set_mass)
    return target_set_total / len(replay_branches), min(owned_shares)


def train_replay_coverage_steps(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    lesson_example: AnswerExample,
    examples: list[AnswerExample],
    *,
    repeat: int,
    rng_seed: int,
    preserve_covered_targets: bool = False,
) -> None:
    lesson = direct_answer_lesson(
        tokenizer,
        model.config.context_size,
        lesson_example,
        ANSWER_TERMINATOR,
    )
    rng = random.Random(rng_seed)
    for _ in range(repeat):
        train_direct_answer_branch_context_replay_coverage_unlikelihood(
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
            preserve_covered_targets=preserve_covered_targets,
        )
