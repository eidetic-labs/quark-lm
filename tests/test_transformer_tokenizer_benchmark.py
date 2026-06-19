from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_tokenizer_benchmark import (
    BENCHMARK_ID,
    run_tokenizer_comparison_benchmark,
)
from tokenizer_artifact_validation import validate_tokenizer_artifacts


class TransformerTokenizerBenchmarkTest(unittest.TestCase):
    def test_subword_tokenizer_reduces_long_answer_tokens_without_full_answer_tokens(self) -> None:
        report = run_tokenizer_comparison_benchmark()

        self.assertEqual(report["benchmark_id"], BENCHMARK_ID)
        self.assertTrue(report["passed"])
        self.assertEqual(report["report"]["full_answer_tokens"], [])
        self.assertTrue(report["report"]["round_trip_ok"])
        validate_tokenizer_artifacts(
            report["manifest"],
            report["report"],
            manifest_hash=report["tokenizer_manifest_hash"],
        )
        self.assertGreater(report["report"]["token_count_savings"], 0)
        long_records = [
            record
            for record in report["records"]
            if record["source"] == "long"
        ]
        self.assertTrue(
            any(record["target_token_savings"] > 0 for record in long_records)
        )


if __name__ == "__main__":
    unittest.main()
