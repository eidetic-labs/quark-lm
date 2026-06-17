from __future__ import annotations

import random

from support.core import (
    ANSWER_TERMINATOR,
    AnswerExample,
    CharTokenizer,
    TinyTransformerLM,
    TransformerConfig,
)
from support.direct_answer import direct_answer_target_balanced_branch_diversity_batch

BranchBatch = list[tuple[list[int], int, int]]


def replay_coverage_fixture() -> tuple[AnswerExample, list[AnswerExample], CharTokenizer]:
    near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
    green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
    tree = AnswerExample(prompt="q: owner?\na:", target=" tree.", source="qa:owner")
    examples = [near, green, tree]
    text = "".join(example.prompt + example.target for example in examples)
    tokenizer = CharTokenizer.train(text + ANSWER_TERMINATOR)
    return near, examples, tokenizer


def initialized_coverage_model(
    tokenizer: CharTokenizer,
    *,
    seed: int,
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
    model.bout[tokenizer.stoi["n"]].data = 5.0
    model.bout[tokenizer.stoi["."]].data = 4.0
    return model


def branch_training_batch(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    lesson_example: AnswerExample,
    examples: list[AnswerExample],
) -> BranchBatch:
    return target_balanced_branch_batch(
        model,
        tokenizer,
        lesson_example,
        examples,
        rng_seed=15,
        batch_size=2,
    )


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
