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
    direct_answer_target_balanced_branch_diversity_batch,
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
