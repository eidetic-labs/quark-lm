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
    build_answer_selector,
    direct_answer_lesson,
    evaluate_answer_records,
    evaluate_direct_answer_records,
    train_direct_answer_first_error,
)


class TransformerDirectAnswerEvaluationTest(unittest.TestCase):
    def test_direct_answer_eval_reports_strict_exact_without_candidates(self) -> None:
        record = {
            "id": "one",
            "prompt": "q:\na:",
            "target": " a.",
        }
        tokenizer = CharTokenizer.train(
            record["prompt"] + record["target"] + ANSWER_TERMINATOR
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=4,
                embedding_dim=3,
                feedforward_dim=5,
                seed=25,
            )
        )
        example = AnswerExample(
            prompt=record["prompt"],
            target=record["target"],
            source="qa:color",
        )
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            example,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(4)
        for _ in range(120):
            train_direct_answer_first_error(
                model,
                tokenizer,
                example,
                lesson,
                rng,
                learning_rate=0.12,
                terminator=ANSWER_TERMINATOR,
            )

        result = evaluate_direct_answer_records(
            model,
            tokenizer,
            [record],
            max_new_chars=16,
            terminator=ANSWER_TERMINATOR,
        )

        self.assertEqual(result["exact"], 1)
        self.assertEqual(result["failed_records"], [])

    def test_answer_eval_can_use_candidate_selector(self) -> None:
        records, examples = _selector_records_and_examples()
        selector = build_answer_selector(examples, seed=22)
        for _ in range(80):
            for example in examples:
                selector.train_step(
                    example,
                    learning_rate=0.08,
                    candidates=[" green.", " in the box."],
                )
        tokenizer = CharTokenizer.train(
            records[0]["prompt"] + " green." + " in the box."
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=12,
                embedding_dim=4,
                feedforward_dim=8,
                seed=23,
            )
        )

        result = evaluate_answer_records(
            model,
            tokenizer,
            records,
            candidates=[" in the box.", " green."],
            max_new_chars=12,
            include_completions=False,
            selector=selector,
        )

        self.assertEqual(result["candidate"], 1)
        self.assertEqual(result["failed_candidate_records"], [])

    def test_answer_eval_can_emit_selector_choice_as_completion(self) -> None:
        records, examples = _selector_records_and_examples()
        selector = build_answer_selector(examples, seed=24)
        candidates = [" in the box.", " green."]
        for _ in range(80):
            for example in examples:
                selector.train_step(example, learning_rate=0.08, candidates=candidates)
        tokenizer = CharTokenizer.train(records[0]["prompt"] + "".join(candidates))
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=12,
                embedding_dim=4,
                feedforward_dim=8,
                seed=25,
            )
        )

        result = evaluate_answer_records(
            model,
            tokenizer,
            records,
            candidates=candidates,
            max_new_chars=12,
            include_completions=False,
            selector=selector,
            emit_selected_candidate=True,
        )

        self.assertEqual(result["exact"], 1)
        self.assertEqual(result["candidate"], 1)
        self.assertEqual(result["failed_records"], [])
        self.assertEqual(result["failed_candidate_records"], [])


def _selector_records_and_examples() -> tuple[list[dict[str, str]], list[AnswerExample]]:
    records = [
        {
            "id": "one",
            "prompt": "question: what color is mia's ring?\nanswer:",
            "target": " green.",
        }
    ]
    examples = [
        AnswerExample(
            prompt=records[0]["prompt"],
            target=" green.",
            source="qa:color",
        ),
        AnswerExample(
            prompt="question: where is mia's ring?\nanswer:",
            target=" in the box.",
            source="qa:place",
        ),
    ]
    return records, examples


if __name__ == "__main__":
    unittest.main()
