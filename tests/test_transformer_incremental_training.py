from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from support.core import TinyTransformerLM, continuation_nll
from support.incremental_training import (
    record,
    train_args,
    train_incremental_candidate,
    write_text,
)
from transformer_char_model import train_transformer
from transformer_incremental_update import guarded_incremental_update


class TransformerIncrementalTrainingTest(unittest.TestCase):
    def test_resume_training_extends_tokenizer_and_persists_learned_lesson(self) -> None:
        prompt = "q2:\na:"
        target = " ok!"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base_text = "q:\na: no\n" * 4
            expanded_text = base_text + ("q2:\na: ok!\n" * 4)
            base_corpus = write_text(root / "base.txt", base_text)
            base_valid = write_text(root / "base_valid.txt", base_text)
            expanded_corpus = write_text(root / "expanded.txt", expanded_text)
            expanded_valid = write_text(root / "expanded_valid.txt", expanded_text)

            base_metrics = train_transformer(
                train_args(
                    corpus=base_corpus,
                    valid=base_valid,
                    run=root / "base_run",
                    steps=0,
                )
            )
            base_model, base_tokenizer = TinyTransformerLM.load(
                Path(base_metrics["checkpoint"])
            )
            if base_tokenizer is None:
                self.fail("base checkpoint did not include tokenizer")

            resumed_metrics = train_transformer(
                train_args(
                    corpus=expanded_corpus,
                    valid=expanded_valid,
                    run=root / "resumed_run",
                    steps=640,
                    resume_checkpoint=Path(base_metrics["checkpoint"]),
                )
            )
            learned_model, learned_tokenizer = TinyTransformerLM.load(
                Path(resumed_metrics["checkpoint"])
            )

        if learned_tokenizer is None:
            self.fail("resumed checkpoint did not include tokenizer")
        before_tokenizer = base_tokenizer.extend(expanded_text)
        base_model.resize_vocab(before_tokenizer.vocab_size)
        before = continuation_nll(base_model, before_tokenizer, prompt, target)
        after = continuation_nll(learned_model, learned_tokenizer, prompt, target)

        self.assertTrue(learned_tokenizer.extends(base_tokenizer))
        self.assertEqual(resumed_metrics["resume"]["tokenizer_extended"], True)
        self.assertEqual(resumed_metrics["resume"]["added_tokens"], ["!", "2", "k"])
        self.assertGreater(before, after)
        self.assertEqual(learned_model.generate(learned_tokenizer, prompt, len(target)), target)

    def test_guarded_incremental_update_accepts_passing_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base_metrics, candidate_metrics = train_incremental_candidate(
                root,
                old_repeats=12,
                new_repeats=4,
                candidate_steps=1500,
            )
            accepted_checkpoint = root / "accepted" / "transformer.json"
            report_path = root / "reports" / "accepted_update.json"

            report = guarded_incremental_update(
                model_cls=TinyTransformerLM,
                base_checkpoint=Path(base_metrics["checkpoint"]),
                candidate_checkpoint=Path(candidate_metrics["checkpoint"]),
                accepted_checkpoint=accepted_checkpoint,
                new_lesson_records=[record("new", "q2:\na:", " ok!")],
                regression_records=[record("old", "q:\na:", " no")],
                report_path=report_path,
            )
            accepted_model, accepted_tokenizer = TinyTransformerLM.load(accepted_checkpoint)
            written_report = _read_json(report_path)

            if accepted_tokenizer is None:
                self.fail("accepted checkpoint did not include tokenizer")
            self.assertTrue(report["accepted"])
            self.assertEqual(report["status"], "accepted")
            self.assertEqual(report["rejection_reasons"], [])
            self.assertEqual(written_report, report)
            self.assertTrue(accepted_checkpoint.exists())
            self.assertEqual(
                accepted_model.generate(accepted_tokenizer, "q2:\na:", 4),
                " ok!",
            )
            self.assertEqual(
                accepted_model.generate(accepted_tokenizer, "q:\na:", 3),
                " no",
            )

    def test_guarded_incremental_update_rejects_regression_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base_metrics, candidate_metrics = train_incremental_candidate(
                root,
                old_repeats=16,
                new_repeats=8,
                candidate_steps=1000,
            )
            accepted_checkpoint = root / "accepted" / "transformer.json"
            report_path = root / "reports" / "rejected_update.json"

            report = guarded_incremental_update(
                model_cls=TinyTransformerLM,
                base_checkpoint=Path(base_metrics["checkpoint"]),
                candidate_checkpoint=Path(candidate_metrics["checkpoint"]),
                accepted_checkpoint=accepted_checkpoint,
                new_lesson_records=[record("new", "q2:\na:", " ok!")],
                regression_records=[record("old", "q:\na:", " no")],
                report_path=report_path,
            )
            written_report = _read_json(report_path)

            self.assertFalse(report["accepted"])
            self.assertEqual(report["status"], "rejected")
            self.assertFalse(accepted_checkpoint.exists())
            self.assertIsNone(report["accepted_checkpoint"])
            self.assertEqual(written_report, report)
            self.assertIn(
                "regression_target_nll_preserved",
                report["rejection_reasons"],
            )


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
