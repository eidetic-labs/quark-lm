from __future__ import annotations

import unittest

from support.branch_diversity import direct_answer_branch_profile
from support.core import (
    ANSWER_TERMINATOR,
    AnswerExample,
    CharTokenizer,
    TinyTransformerLM,
    TransformerConfig,
)
from support.direct_answer import direct_answer_first_error


class TransformerBranchProfilesTest(unittest.TestCase):
    def test_direct_answer_first_error_targets_greedy_mismatch(self) -> None:
        example = AnswerExample(prompt="q:\na:", target=" a.", source="qa:color")
        tokenizer = CharTokenizer.train(example.prompt + example.target + ANSWER_TERMINATOR)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=4,
                embedding_dim=3,
                feedforward_dim=5,
                seed=24,
            )
        )
        wrong_id = tokenizer.stoi["."]
        model.bout[wrong_id].data = 5.0

        repair = direct_answer_first_error(
            model,
            tokenizer,
            example,
            ANSWER_TERMINATOR,
        )

        self.assertIsNotNone(repair)
        _context, target_id, predicted_id, position = repair
        self.assertEqual(tokenizer.itos[target_id], " ")
        self.assertEqual(tokenizer.itos[predicted_id], ".")
        self.assertEqual(position, 0)

    def test_direct_answer_branch_profile_summarizes_branch_confusion(self) -> None:
        record = {
            "id": "color",
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
                seed=24,
            )
        )
        wrong_id = tokenizer.stoi["."]
        model.bout[wrong_id].data = 5.0

        profile = direct_answer_branch_profile(
            model,
            tokenizer,
            [record],
            branch_position=0,
            terminator=ANSWER_TERMINATOR,
        )

        self.assertEqual(profile["count"], 1)
        self.assertEqual(profile["correct"], 0)
        self.assertEqual(profile["skipped"], 0)
        self.assertLess(profile["avg_target_margin"], 0.0)
        self.assertGreater(profile["target_rank"]["avg"], 1.0)
        self.assertEqual(profile["target_rank"]["top1_rate"], 0.0)
        self.assertEqual(profile["target_rank"]["vocab_size"], tokenizer.vocab_size)
        self.assertEqual(profile["target_tokens"][0], {"value": " ", "count": 1})
        self.assertEqual(profile["predicted_tokens"][0], {"value": ".", "count": 1})
        self.assertEqual(profile["confusions"][0], {"value": "' '->'.'", "count": 1})
        self.assertEqual(profile["failed_records"][0]["id"], "color")
        self.assertEqual(profile["failed_records"][0]["target_token"], " ")
        self.assertEqual(profile["failed_records"][0]["predicted_token"], ".")
        self.assertGreaterEqual(profile["failed_records"][0]["target_rank"], 2)
        self.assertEqual(
            profile["failed_records"][0]["top_predictions"][0]["token"],
            ".",
        )

    def test_direct_answer_branch_profile_reports_diversity_collapse(self) -> None:
        records = [
            {"id": "near", "prompt": "q: where?\na:", "target": " near."},
            {"id": "green", "prompt": "q: color?\na:", "target": " green."},
        ]
        tokenizer = CharTokenizer.train(
            "".join(record["prompt"] + record["target"] for record in records)
            + ANSWER_TERMINATOR
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=43,
            )
        )
        wrong_id = tokenizer.stoi["."]
        model.bout[wrong_id].data = 5.0

        profile = direct_answer_branch_profile(
            model,
            tokenizer,
            records,
            branch_position=1,
            terminator=ANSWER_TERMINATOR,
        )

        self.assertEqual(profile["diversity"]["target_unique"], 2)
        self.assertEqual(profile["diversity"]["predicted_unique"], 1)
        self.assertEqual(profile["diversity"]["target_token_coverage"], 0.0)
        self.assertEqual(profile["diversity"]["dominant_predicted_token"], ".")
        self.assertEqual(profile["diversity"]["dominant_predicted_count"], 2)
        self.assertEqual(profile["diversity"]["dominant_predicted_rate"], 1.0)
        self.assertTrue(profile["diversity"]["collapsed"])
        self.assertEqual(
            profile["diversity"]["missing_target_tokens"],
            [{"value": "n", "count": 1}, {"value": "g", "count": 1}],
        )


if __name__ == "__main__":
    unittest.main()
