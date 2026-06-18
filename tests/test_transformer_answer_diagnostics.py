from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tokenizer import CharTokenizer
from transformer_answer_diagnostics import answer_diagnostics, first_drift_index
from transformer_model import TransformerConfig
from transformer_tiny_lm import TinyTransformerLM


class TransformerAnswerDiagnosticsTest(unittest.TestCase):
    def test_first_drift_index_reports_first_changed_character(self) -> None:
        self.assertIsNone(first_drift_index(" kitchen.", " kitchen."))
        self.assertEqual(first_drift_index(" kitchen.", " kitehen."), 4)
        self.assertEqual(first_drift_index(" kitchen.", " kit"), 4)

    def test_answer_diagnostics_records_token_nll_by_position(self) -> None:
        tokenizer = CharTokenizer.train("q place mia ball\nA: kitchen.")
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=12,
                embedding_dim=4,
                feedforward_dim=8,
                seed=5,
            )
        )

        report = answer_diagnostics(model, tokenizer, "q place mia ball\nA:", " kitchen.")

        self.assertEqual(report["target_token_count"], len(tokenizer.encode(" kitchen.")))
        self.assertEqual(len(report["per_token_nll"]), report["target_token_count"])
        self.assertIn("first_drift_index", report)


if __name__ == "__main__":
    unittest.main()
