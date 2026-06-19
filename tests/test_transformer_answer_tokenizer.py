from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from support.commands import parse_args, train_transformer_answers
from tokenizer_artifact_validation import validate_tokenizer_artifacts


class TransformerAnswerTokenizerTest(unittest.TestCase):
    def test_answer_train_can_use_closed_world_subword_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "answer-subword-screen"
            args = parse_args(
                [
                    "answer-train",
                    "--run",
                    str(run_dir),
                    "--steps",
                    "0",
                    "--eval-every",
                    "0",
                    "--direct-answer-steps",
                    "0",
                    "--selector-steps",
                    "0",
                    "--generator-steps",
                    "0",
                    "--candidate-scope",
                    "eval",
                    "--skip-post-direct-snapshot",
                    "--embedding-dim",
                    "4",
                    "--feedforward-dim",
                    "8",
                    "--context-size",
                    "8",
                    "--tokenizer",
                    "closed-world-subword",
                    "--tokenizer-max-token-chars",
                    "4",
                    "--tokenizer-max-new-tokens",
                    "8",
                ]
            )

            metrics = train_transformer_answers(args)

            manifest = json.loads(
                (run_dir / "tokenizer_manifest.json").read_text(encoding="utf-8")
            )
            report = json.loads(
                (run_dir / "tokenizer_report.json").read_text(encoding="utf-8")
            )
            diagnostics = json.loads(
                Path(metrics["long_answer_diagnostics"]["path"]).read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(metrics["tokenizer_type"], "closed-world-subword")
        self.assertEqual(manifest["tokenizer_type"], "closed-world-subword")
        self.assertEqual(report["tokenizer_type"], "closed-world-subword")
        validate_tokenizer_artifacts(
            manifest,
            report,
            manifest_hash=metrics["tokenizer_manifest_hash"],
        )
        self.assertEqual(
            metrics["tokenizer_manifest_hash"],
            metrics["sweep_plan"]["current_trial"]["tokenizer_manifest_hash"],
        )
        self.assertEqual(
            metrics["sweep_plan"]["current_trial"]["tokenizer_type"],
            "closed-world-subword",
        )
        self.assertEqual(diagnostics["kind"], "transformer_long_answer_diagnostics")
        self.assertFalse(metrics["pretrained_tokenizer"])


if __name__ == "__main__":
    unittest.main()
