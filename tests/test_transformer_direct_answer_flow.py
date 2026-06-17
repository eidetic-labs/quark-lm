from __future__ import annotations

import random
import unittest

from support.core import (
    ANSWER_TERMINATOR,
    AnswerExample,
    CharTokenizer,
    TinyTransformerLM,
    TransformerConfig,
)
from support.direct_answer import (
    direct_answer_lesson,
    direct_answer_sequence_nll,
    train_direct_answer_lesson,
)


class TransformerDirectAnswerFlowTest(unittest.TestCase):
    def test_direct_answer_training_updates_transformer_without_candidates(self) -> None:
        example = AnswerExample(
            prompt="question: what color is mia's ring?\nanswer:",
            target=" green.",
            source="qa:color",
        )
        tokenizer = CharTokenizer.train(example.prompt + example.target + ANSWER_TERMINATOR)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=12,
                embedding_dim=4,
                feedforward_dim=8,
                seed=24,
            )
        )
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            example,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(3)

        before = direct_answer_sequence_nll(model, tokenizer, example, ANSWER_TERMINATOR)
        for _ in range(120):
            train_direct_answer_lesson(model, lesson, rng, learning_rate=0.05)
        after = direct_answer_sequence_nll(model, tokenizer, example, ANSWER_TERMINATOR)

        self.assertGreater(before, after)


if __name__ == "__main__":
    unittest.main()
