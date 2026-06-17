"""Direct-answer repair fixtures used by transformer repair tests."""

from __future__ import annotations

from dataclasses import dataclass

from support.core import (
    ANSWER_TERMINATOR,
    AnswerExample,
    CharTokenizer,
    TinyTransformerLM,
    TransformerConfig,
)
from support.direct_answer import direct_answer_lesson


@dataclass(frozen=True)
class DirectAnswerRepairFixture:
    example: AnswerExample
    tokenizer: CharTokenizer
    model: TinyTransformerLM

    @property
    def lesson(self) -> list[tuple[list[int], int]]:
        return direct_answer_lesson(
            self.tokenizer,
            self.model.config.context_size,
            self.example,
            ANSWER_TERMINATOR,
        )


def direct_answer_repair_fixture(
    *,
    target: str,
    source: str,
    seed: int,
    prompt: str = "q:\na:",
    context_size: int = 6,
    embedding_dim: int = 3,
    feedforward_dim: int = 5,
) -> DirectAnswerRepairFixture:
    example = AnswerExample(prompt=prompt, target=target, source=source)
    tokenizer = CharTokenizer.train(example.prompt + example.target + ANSWER_TERMINATOR)
    model = TinyTransformerLM.init_random(
        TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=context_size,
            embedding_dim=embedding_dim,
            feedforward_dim=feedforward_dim,
            seed=seed,
        )
    )
    return DirectAnswerRepairFixture(
        example=example,
        tokenizer=tokenizer,
        model=model,
    )


__all__ = ["DirectAnswerRepairFixture", "direct_answer_repair_fixture"]
