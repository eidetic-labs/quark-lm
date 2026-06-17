from __future__ import annotations

import math
import random

from support.core import (
    ANSWER_TERMINATOR,
    AnswerExample,
    CharTokenizer,
    TinyTransformerLM,
    TransformerConfig,
)
from support.direct_answer import (
    direct_answer_lesson,
    direct_answer_target_balanced_branch_diversity_batch,
    train_direct_answer_branch_bidirectional_binding_unlikelihood,
    train_direct_answer_branch_coverage_binding_unlikelihood,
    train_direct_answer_branch_rank_margin_unlikelihood,
)

BranchBatch = list[tuple[list[int], int, int]]


def branch_binding_fixture() -> tuple[AnswerExample, list[AnswerExample], CharTokenizer]:
    near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
    green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
    tree = AnswerExample(prompt="q: owner?\na:", target=" tree.", source="qa:owner")
    examples = [near, green, tree]
    text = "".join(example.prompt + example.target for example in examples)
    tokenizer = CharTokenizer.train(text + ANSWER_TERMINATOR)
    return near, examples, tokenizer


def initialized_branch_binding_model(
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
    batch_size: int = 3,
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


def train_rank_margin_steps(
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
        train_direct_answer_branch_rank_margin_unlikelihood(
            model,
            tokenizer,
            lesson_example,
            examples,
            lesson,
            rng,
            learning_rate=0.03,
            negative_weight=1.0,
            positive_weight=1.0,
            margin_weight=2.0,
            branch_position=1,
            batch_size=3,
            hard_negative_count=5,
            terminator=ANSWER_TERMINATOR,
            balance_targets=True,
        )


def train_bidirectional_binding_steps(
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
        train_direct_answer_branch_bidirectional_binding_unlikelihood(
            model,
            tokenizer,
            lesson_example,
            examples,
            lesson,
            rng,
            learning_rate=0.03,
            negative_weight=1.0,
            positive_weight=1.0,
            binding_weight=2.0,
            branch_position=1,
            batch_size=3,
            terminator=ANSWER_TERMINATOR,
            balance_targets=True,
        )


def train_coverage_binding_steps(
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
        train_direct_answer_branch_coverage_binding_unlikelihood(
            model,
            tokenizer,
            lesson_example,
            examples,
            lesson,
            rng,
            learning_rate=0.03,
            negative_weight=1.0,
            positive_weight=1.0,
            binding_weight=2.0,
            branch_position=1,
            batch_size=3,
            hard_negative_count=5,
            terminator=ANSWER_TERMINATOR,
            balance_targets=True,
        )
