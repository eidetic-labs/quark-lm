from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from answer_model import AnswerExample
from self_improvement_tokenizer import (
    build_self_improvement_tokenizer_candidate,
    protected_answer_texts,
    tokenizer_candidate_guard,
    tokenizer_candidate_record,
)


class SelfImprovementTokenizerTest(unittest.TestCase):
    def test_build_candidate_writes_guarded_manifest_and_report(self) -> None:
        examples = [
            AnswerExample("q:\na:", " no.", "qa"),
            AnswerExample("q2:\na:", " ok.", "qa"),
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train_text = root / "train.txt"
            train_text.write_text("q:\na: no.\nq2:\na: ok.\nq3:\na: note.\n", encoding="utf-8")

            candidate = build_self_improvement_tokenizer_candidate(
                train_text,
                examples,
                root / "tokenizer_manifest.json",
                root / "tokenizer_report.json",
            )

            manifest = json.loads((root / "tokenizer_manifest.json").read_text())
            report = json.loads((root / "tokenizer_report.json").read_text())

        guard = tokenizer_candidate_guard(candidate)
        self.assertEqual(candidate["status"], "candidate_generated")
        self.assertEqual(manifest["tokenizer_type"], "closed-world-subword")
        self.assertTrue(report["round_trip_ok"])
        self.assertEqual(candidate["summary"]["full_answer_token_count"], 0)
        self.assertTrue(guard["passed"])

    def test_protected_answer_texts_ignore_non_answer_shapes(self) -> None:
        examples = [
            AnswerExample("q:\na:", " yes.", "qa"),
            object(),
        ]

        self.assertEqual(protected_answer_texts(examples), {" yes."})

    def test_guard_rejects_full_answer_token_candidate(self) -> None:
        candidate = tokenizer_candidate_record(
            Path("tokenizer_manifest.json"),
            Path("tokenizer_report.json"),
            "abc123",
            {
                "tokenizer_type": "closed-world-subword",
                "corpus_hash": "hash",
                "purity": {
                    "pretrained_tokenizer": False,
                    "external_vocabulary": False,
                    "admitted_corpus_only": True,
                },
                "rejected_candidates": [],
            },
            {
                "round_trip_ok": True,
                "accepted_token_count": 1,
                "token_count_savings": 2,
                "compression_ratio": 0.9,
                "branch_diversity_score": 1.0,
                "full_answer_tokens": [" no."],
            },
        )

        guard = tokenizer_candidate_guard(candidate)

        self.assertFalse(guard["passed"])
        failed = [check["name"] for check in guard["checks"] if not check["passed"]]
        self.assertEqual(failed, ["no_full_answer_tokens"])


if __name__ == "__main__":
    unittest.main()
