from __future__ import annotations

import random
import unittest

from support.core import (
    AnswerExample,
    CharTokenizer,
    TinyTransformerLM,
    TransformerConfig,
    continuation_nll,
)
from support.direct_answer import (
    answer_sequence_nll,
    train_answer_char,
    train_answer_mixed_step,
)


class TransformerAnswerFlowTest(unittest.TestCase):
    def test_answer_lesson_training_reduces_continuation_loss(self) -> None:
        example = AnswerExample(
            prompt="question: where is mia's ring?\nanswer:",
            target=" in the box.",
            source="qa:place",
        )
        tokenizer = CharTokenizer.train(example.prompt + example.target)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=9,
            )
        )
        before = answer_sequence_nll(model, tokenizer, example)
        rng = random.Random(1)
        for _ in range(80):
            train_answer_char(model, tokenizer, example, rng, learning_rate=0.05)
        after = answer_sequence_nll(model, tokenizer, example)

        self.assertGreater(before, after)

    def test_mixed_answer_training_improves_candidate_margin(self) -> None:
        example = AnswerExample(
            prompt="question: what color is mia's ring?\nanswer:",
            target=" green.",
            source="qa:color",
        )
        negative = " red."
        tokenizer = CharTokenizer.train(example.prompt + example.target + negative)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=12,
                embedding_dim=4,
                feedforward_dim=8,
                seed=13,
            )
        )
        rng = random.Random(2)

        before_margin = continuation_nll(
            model,
            tokenizer,
            example.prompt,
            example.target,
        ) - continuation_nll(model, tokenizer, example.prompt, negative)
        for _ in range(60):
            train_answer_mixed_step(
                model,
                tokenizer,
                example,
                rng,
                learning_rate=0.05,
                candidates=[example.target, negative],
                target_loss_weight=1.0,
                choice_loss_weight=1.0,
                choice_negatives=1,
                choice_max_chars=4,
            )
        after_margin = continuation_nll(
            model,
            tokenizer,
            example.prompt,
            example.target,
        ) - continuation_nll(model, tokenizer, example.prompt, negative)

        self.assertLess(after_margin, before_margin)


if __name__ == "__main__":
    unittest.main()
