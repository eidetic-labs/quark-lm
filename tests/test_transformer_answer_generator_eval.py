from __future__ import annotations

import unittest

from support.core import AnswerExample, CharTokenizer, TinyTransformerLM, TransformerConfig
from support.direct_answer import (
    TransformerGuidedAnswerGenerator,
    build_transformer_answer_generator,
    evaluate_answer_generator_records,
)


class TransformerAnswerGeneratorEvalTest(unittest.TestCase):
    def test_transformer_guided_generator_learns_without_candidates(self) -> None:
        examples = _answer_generator_examples()
        tokenizer = CharTokenizer.train(
            "".join(example.prompt + example.target for example in examples)
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=12,
                embedding_dim=4,
                feedforward_dim=8,
                seed=26,
            )
        )
        generator = build_transformer_answer_generator(
            examples,
            model,
            tokenizer,
            seed=27,
            max_answer_chars=24,
            transformer_top_k=2,
        )

        before = generator.sequence_loss(
            model,
            tokenizer,
            examples[0].prompt,
            examples[0].target,
        )
        for _ in range(180):
            for example in examples:
                generator.train_example(model, tokenizer, example, learning_rate=0.08)
        after = generator.sequence_loss(
            model,
            tokenizer,
            examples[0].prompt,
            examples[0].target,
        )

        self.assertIsInstance(generator, TransformerGuidedAnswerGenerator)
        self.assertGreater(before, after)
        self.assertEqual(generator.generate(model, tokenizer, examples[0].prompt), " green.")
        self.assertEqual(generator.generate(model, tokenizer, examples[1].prompt), " in the box.")

    def test_answer_generator_eval_reports_exact_without_candidates(self) -> None:
        examples = [
            AnswerExample(
                prompt="question: what color is mia's ring?\nanswer:",
                target=" green.",
                source="qa:color",
            )
        ]
        records = [
            {
                "id": "one",
                "prompt": examples[0].prompt,
                "target": examples[0].target,
            }
        ]
        tokenizer = CharTokenizer.train(examples[0].prompt + examples[0].target)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=12,
                embedding_dim=4,
                feedforward_dim=8,
                seed=28,
            )
        )
        generator = build_transformer_answer_generator(
            examples,
            model,
            tokenizer,
            seed=29,
            max_answer_chars=24,
            transformer_top_k=2,
        )
        for _ in range(180):
            generator.train_example(model, tokenizer, examples[0], learning_rate=0.08)

        result = evaluate_answer_generator_records(generator, model, tokenizer, records)

        self.assertEqual(result["exact"], 1)
        self.assertEqual(result["failed_records"], [])


def _answer_generator_examples() -> list[AnswerExample]:
    return [
        AnswerExample(
            prompt="question: what color is mia's ring?\nanswer:",
            target=" green.",
            source="qa:color",
        ),
        AnswerExample(
            prompt="question: where is mia's ring?\nanswer:",
            target=" in the box.",
            source="qa:place",
        ),
    ]


if __name__ == "__main__":
    unittest.main()
