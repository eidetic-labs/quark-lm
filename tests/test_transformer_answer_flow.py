from __future__ import annotations

import random
import unittest

from transformer_char_model_test_support import (
    ANSWER_TERMINATOR,
    AnswerCandidateSelector,
    AnswerExample,
    CharTokenizer,
    TinyTransformerLM,
    TransformerConfig,
    answer_sequence_nll,
    build_answer_selector,
    continuation_nll,
    direct_answer_lesson,
    direct_answer_sequence_nll,
    evaluate_answer_records,
    sampled_choice_candidates,
    train_answer_char,
    train_answer_mixed_step,
    train_direct_answer_lesson,
    transformer_answer_generator_training_pool,
    transformer_direct_answer_training_pool,
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

    def test_answer_eval_reports_candidate_matches(self) -> None:
        records = [
            {
                "id": "one",
                "prompt": "question: where is mia's ring?\nanswer:",
                "target": " in the box.",
            }
        ]
        tokenizer = CharTokenizer.train(records[0]["prompt"] + records[0]["target"])
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=11,
            )
        )

        result = evaluate_answer_records(
            model,
            tokenizer,
            records,
            candidates=[" in the box."],
            max_new_chars=12,
        )

        self.assertEqual(result["count"], 1)
        self.assertEqual(result["candidate"], 1)
        self.assertEqual(result["failed_candidate_records"], [])

    def test_answer_eval_can_skip_slow_completions(self) -> None:
        records = [
            {
                "id": "one",
                "prompt": "question: where is mia's ring?\nanswer:",
                "target": " in the box.",
            }
        ]
        tokenizer = CharTokenizer.train(records[0]["prompt"] + records[0]["target"])
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=12,
            )
        )

        result = evaluate_answer_records(
            model,
            tokenizer,
            records,
            candidates=[" in the box."],
            max_new_chars=12,
            include_completions=False,
        )

        self.assertIsNone(result["exact"])
        self.assertEqual(result["candidate"], 1)
        self.assertEqual(result["failed_records"], [])

    def test_sampled_choice_candidates_keeps_target_first(self) -> None:
        rng = random.Random(4)

        candidates = sampled_choice_candidates(
            " green.",
            [" red.", " green.", " blue.", " red."],
            rng,
            negative_count=1,
        )

        self.assertEqual(candidates[0], " green.")
        self.assertEqual(len(candidates), 2)
        self.assertNotEqual(candidates[1], " green.")

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

    def test_answer_candidate_selector_learns_from_closed_world_examples(self) -> None:
        examples = [
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
        selector = build_answer_selector(examples, seed=21)
        candidates = [" green.", " in the box."]

        before = selector.loss(examples[0].prompt, examples[0].target, candidates)
        for _ in range(80):
            for example in examples:
                selector.train_step(example, learning_rate=0.08, candidates=candidates)
        after = selector.loss(examples[0].prompt, examples[0].target, candidates)

        self.assertIsInstance(selector, AnswerCandidateSelector)
        self.assertGreater(before, after)
        self.assertEqual(selector.predict(examples[0].prompt, candidates), " green.")
        self.assertEqual(selector.predict(examples[1].prompt, candidates), " in the box.")

    def test_transformer_generator_pool_prioritizes_long_operational_lessons(self) -> None:
        fact = AnswerExample(
            prompt="question: what color is mia's ring?\nanswer:",
            target=" green.",
            source="qa:color",
        )
        learning = AnswerExample(
            prompt="question: how do you improve?\nanswer:",
            target=" by admitted training data.",
            source="qa:learning",
        )

        pool = transformer_answer_generator_training_pool([fact, learning])

        self.assertGreater(pool.count(learning), pool.count(fact))

    def test_direct_answer_pool_prioritizes_long_operational_lessons(self) -> None:
        fact = AnswerExample(
            prompt="question: what color is mia's ring?\nanswer:",
            target=" green.",
            source="qa:color",
        )
        learning = AnswerExample(
            prompt="question: how do you improve?\nanswer:",
            target=" by admitted training data.",
            source="qa:learning",
        )

        pool = transformer_direct_answer_training_pool([fact, learning])

        self.assertGreater(pool.count(learning), pool.count(fact))

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
