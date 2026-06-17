from __future__ import annotations

import unittest

from support.core import CharTokenizer, TinyTransformerLM, TransformerConfig
from support.direct_answer import evaluate_answer_records


class TransformerAnswerEvalTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
