from __future__ import annotations

import unittest

from support.core import ANSWER_TERMINATOR, CharTokenizer, TinyTransformerLM, TransformerConfig
from support.direct_answer import (
    audit_direct_answer_branch_context_coverage,
    summarize_branch_context_coverage_gate,
)


class TransformerBranchContextCoverageTest(unittest.TestCase):
    def test_branch_context_coverage_marks_truncated_semantic_branch(self) -> None:
        record = {
            "id": "place",
            "prompt": "question: where is mia's ball?\nanswer:",
            "target": " under.",
        }
        tokenizer = CharTokenizer.train(
            record["prompt"] + record["target"] + ANSWER_TERMINATOR
        )
        narrow = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=3,
                feedforward_dim=5,
                seed=41,
            )
        )
        wide = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=48,
                embedding_dim=3,
                feedforward_dim=5,
                seed=41,
            )
        )

        narrow_audit = audit_direct_answer_branch_context_coverage(
            narrow,
            tokenizer,
            [record],
            branch_position=1,
            terminator=ANSWER_TERMINATOR,
        )
        wide_audit = audit_direct_answer_branch_context_coverage(
            wide,
            tokenizer,
            [record],
            branch_position=1,
            terminator=ANSWER_TERMINATOR,
        )

        self.assertEqual(narrow_audit["semantic_records"], 1)
        self.assertEqual(narrow_audit["missing"], 1)
        self.assertIn(
            "intent:place",
            narrow_audit["missing_records"][0]["missing_features"],
        )
        self.assertEqual(wide_audit["covered"], 1)
        self.assertEqual(wide_audit["missing_records"], [])

    def test_branch_context_coverage_marks_ambiguous_context_collisions(self) -> None:
        records = [
            {"id": "one", "prompt": "q: one\na:", "target": " red."},
            {"id": "two", "prompt": "q: two\na:", "target": " blue."},
        ]
        text = "".join(record["prompt"] + record["target"] for record in records)
        tokenizer = CharTokenizer.train(text + ANSWER_TERMINATOR)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=3,
                embedding_dim=3,
                feedforward_dim=5,
                seed=42,
            )
        )

        audit = audit_direct_answer_branch_context_coverage(
            model,
            tokenizer,
            records,
            branch_position=1,
            terminator=ANSWER_TERMINATOR,
        )

        self.assertEqual(audit["count"], 2)
        self.assertEqual(audit["unique_contexts"], 1)
        self.assertEqual(audit["collision_contexts"], 1)
        self.assertEqual(audit["ambiguous_contexts"], 1)
        self.assertEqual(audit["max_context_reuse"], 2)
        self.assertEqual(audit["max_target_options"], 2)
        self.assertEqual(audit["ambiguous_records"][0]["context_text"], "a: ")
        self.assertEqual(
            audit["ambiguous_records"][0]["target_tokens"],
            [{"value": "r", "count": 1}, {"value": "b", "count": 1}],
        )

    def test_branch_context_coverage_gate_summarizes_blockers(self) -> None:
        summary = summarize_branch_context_coverage_gate(
            {
                "qa": {
                    "count": 2,
                    "semantic_records": 2,
                    "covered": 1,
                    "missing": 1,
                    "covered_rate": 0.5,
                    "ambiguous_contexts": 1,
                    "collision_contexts": 1,
                    "skipped": 0,
                },
                "self": {
                    "count": 1,
                    "semantic_records": 1,
                    "covered": 1,
                    "missing": 0,
                    "covered_rate": 1.0,
                    "ambiguous_contexts": 0,
                    "collision_contexts": 0,
                    "skipped": 0,
                },
            }
        )

        self.assertFalse(summary["passed"])
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["covered"], 2)
        self.assertEqual(summary["missing"], 1)
        self.assertEqual(summary["ambiguous_contexts"], 1)
        self.assertEqual(summary["blocking_evals"][0]["name"], "qa")


if __name__ == "__main__":
    unittest.main()
