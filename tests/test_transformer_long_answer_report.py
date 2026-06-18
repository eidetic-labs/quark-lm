from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tokenizer import CharTokenizer
from transformer_long_answer_report import (
    build_long_answer_diagnostics_report,
    write_long_answer_diagnostics_report,
)
from transformer_model import GenerationConfig, TransformerConfig
from transformer_tiny_lm import TinyTransformerLM


class TransformerLongAnswerReportTest(unittest.TestCase):
    def test_report_records_drift_loss_timing_and_candidate_rank(self) -> None:
        tokenizer = CharTokenizer.train("q color mia ball\nA: blue green.")
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=12,
                embedding_dim=4,
                feedforward_dim=8,
                seed=11,
            )
        )
        eval_records = {
            "qa": [
                {
                    "id": "short",
                    "prompt": "q color mia ball\nA:",
                    "target": " blue.",
                },
                {
                    "id": "long",
                    "prompt": "q color mia ball\nA:",
                    "target": " blue green.",
                },
            ]
        }

        report = build_long_answer_diagnostics_report(
            run_id="run-001",
            model=model,
            tokenizer=tokenizer,
            eval_records=eval_records,
            eval_candidates={"qa": [" blue.", " blue green."]},
            generation_config=GenerationConfig(),
            train_time_seconds=1.25,
            direct_answer_metrics=None,
        )

        record = report["records"][0]
        self.assertEqual(report["kind"], "transformer_long_answer_diagnostics")
        self.assertEqual(record["id"], "long")
        self.assertEqual(record["target_token_count"], len(tokenizer.encode(" blue green.")))
        self.assertEqual(len(record["per_token_nll"]), record["target_token_count"])
        self.assertIn("first_drift_index", record)
        self.assertGreaterEqual(record["generation_time_ms"], 0.0)
        self.assertEqual(record["candidate_ranking"]["candidate_count"], 2)
        self.assertEqual(report["summary"]["train_time_seconds"], 1.25)

    def test_report_writer_serializes_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "long_answer_diagnostics.json"
            write_long_answer_diagnostics_report(
                path,
                {"kind": "transformer_long_answer_diagnostics"},
            )
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["kind"], "transformer_long_answer_diagnostics")


if __name__ == "__main__":
    unittest.main()
