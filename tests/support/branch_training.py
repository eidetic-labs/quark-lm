"""Branch-training fixtures used by transformer branch objective tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from support.core import (
    ANSWER_TERMINATOR,
    AnswerExample,
    CharTokenizer,
    TinyTransformerLM,
    TransformerConfig,
)


@dataclass(frozen=True)
class BranchTrainingFixture:
    near: AnswerExample
    green: AnswerExample
    tree: AnswerExample
    tokenizer: CharTokenizer
    model: TinyTransformerLM

    @property
    def examples(self) -> list[AnswerExample]:
        return [self.near, self.green, self.tree]

    @property
    def records(self) -> list[dict[str, str]]:
        return [
            {"id": "near", "prompt": self.near.prompt, "target": self.near.target},
            {
                "id": "green",
                "prompt": self.green.prompt,
                "target": self.green.target,
            },
            {"id": "tree", "prompt": self.tree.prompt, "target": self.tree.target},
        ]


def branch_training_example_set() -> tuple[AnswerExample, AnswerExample, AnswerExample]:
    return (
        AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place"),
        AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color"),
        AnswerExample(prompt="q: owner?\na:", target=" tree.", source="qa:owner"),
    )


def branch_training_tokenizer(examples: Sequence[AnswerExample]) -> CharTokenizer:
    return CharTokenizer.train(
        "".join(example.prompt + example.target for example in examples)
        + ANSWER_TERMINATOR
    )


def branch_training_fixture(
    *,
    seed: int,
    context_size: int = 8,
    embedding_dim: int = 4,
    feedforward_dim: int = 8,
    extra_examples: Sequence[AnswerExample] | None = None,
) -> BranchTrainingFixture:
    near, green, tree = branch_training_example_set()
    examples = [near, green, tree]
    if extra_examples is not None:
        examples.extend(extra_examples)
    tokenizer = branch_training_tokenizer(examples)
    model = TinyTransformerLM.init_random(
        TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=context_size,
            embedding_dim=embedding_dim,
            feedforward_dim=feedforward_dim,
            seed=seed,
        )
    )
    return BranchTrainingFixture(
        near=near,
        green=green,
        tree=tree,
        tokenizer=tokenizer,
        model=model,
    )


__all__ = [
    "BranchTrainingFixture",
    "branch_training_example_set",
    "branch_training_fixture",
    "branch_training_tokenizer",
]
