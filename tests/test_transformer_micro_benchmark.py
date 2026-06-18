from __future__ import annotations

import unittest

from support.core import TinyTransformerLM
from transformer_micro_benchmark import (
    BENCHMARK_ID,
    run_meaningful_micro_benchmark,
)


class TransformerMicroBenchmarkTest(unittest.TestCase):
    def test_meaningful_micro_corpus_benchmark_trains_heldout_prompt_style(self) -> None:
        report = run_meaningful_micro_benchmark(TinyTransformerLM)

        self.assertEqual(report["benchmark_id"], BENCHMARK_ID)
        self.assertTrue(report["passed"])
        self.assertEqual(report["dataset"]["train_examples"], 4)
        self.assertEqual(report["dataset"]["heldout_examples"], 4)
        self.assertEqual(report["dataset"]["prompt_overlap"], [])
        self.assertEqual(report["dataset"]["target_lengths"], [5])
        self.assertEqual(report["dataset"]["max_target_chars"], 5)
        self.assertEqual(report["train"]["summary"]["exact_rate"], 1.0)
        self.assertEqual(report["train"]["summary"]["candidate_rate"], 1.0)
        self.assertEqual(report["heldout"]["summary"]["exact_rate"], 1.0)
        self.assertEqual(report["heldout"]["summary"]["candidate_rate"], 1.0)
        self.assertEqual(
            {record["target"] for record in report["heldout"]["records"]},
            {" box.", " red.", " cup.", " tan."},
        )
        self.assertFalse(report["dataset"]["pretrained_weights"])
        self.assertFalse(report["dataset"]["pretrained_tokenizer"])


if __name__ == "__main__":
    unittest.main()
