from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from support.commands import parse_args
from support.core import TinyTransformerLM, continuation_nll
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
            base_corpus = _write_text(root / "base.txt", base_text)
            base_valid = _write_text(root / "base_valid.txt", base_text)
            expanded_corpus = _write_text(root / "expanded.txt", expanded_text)
            expanded_valid = _write_text(root / "expanded_valid.txt", expanded_text)

            base_metrics = train_transformer(
                _train_args(
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
                _train_args(
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
            base_metrics, candidate_metrics = _train_incremental_candidate(
                root,
                old_repeats=12,
                new_repeats=4,
                candidate_steps=1500,
            )
            accepted_checkpoint = root / "accepted" / "transformer.json"

            report = guarded_incremental_update(
                model_cls=TinyTransformerLM,
                base_checkpoint=Path(base_metrics["checkpoint"]),
                candidate_checkpoint=Path(candidate_metrics["checkpoint"]),
                accepted_checkpoint=accepted_checkpoint,
                new_lesson_records=[_record("new", "q2:\na:", " ok!")],
                regression_records=[_record("old", "q:\na:", " no")],
            )
            accepted_model, accepted_tokenizer = TinyTransformerLM.load(accepted_checkpoint)

            if accepted_tokenizer is None:
                self.fail("accepted checkpoint did not include tokenizer")
            self.assertTrue(report["accepted"])
            self.assertEqual(report["rejection_reasons"], [])
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
            base_metrics, candidate_metrics = _train_incremental_candidate(
                root,
                old_repeats=16,
                new_repeats=8,
                candidate_steps=1000,
            )
            accepted_checkpoint = root / "accepted" / "transformer.json"

            report = guarded_incremental_update(
                model_cls=TinyTransformerLM,
                base_checkpoint=Path(base_metrics["checkpoint"]),
                candidate_checkpoint=Path(candidate_metrics["checkpoint"]),
                accepted_checkpoint=accepted_checkpoint,
                new_lesson_records=[_record("new", "q2:\na:", " ok!")],
                regression_records=[_record("old", "q:\na:", " no")],
            )

            self.assertFalse(report["accepted"])
            self.assertFalse(accepted_checkpoint.exists())
            self.assertIn(
                "regression_target_nll_preserved",
                report["rejection_reasons"],
            )


def _write_text(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _train_args(
    *,
    corpus: Path,
    valid: Path,
    run: Path,
    steps: int,
    resume_checkpoint: Path | None = None,
) -> argparse.Namespace:
    argv = [
        "train",
        "--corpus",
        str(corpus),
        "--valid",
        str(valid),
        "--run",
        str(run),
        "--steps",
        str(steps),
        "--learning-rate",
        "0.08",
        "--eval-every",
        "0",
        "--valid-limit",
        "128",
        "--context-size",
        "8",
        "--embedding-dim",
        "4",
        "--feedforward-dim",
        "8",
        "--seed",
        "1",
    ]
    if resume_checkpoint is not None:
        argv.extend(["--resume-checkpoint", str(resume_checkpoint)])
    return parse_args(argv)


def _train_incremental_candidate(
    root: Path,
    *,
    old_repeats: int,
    new_repeats: int,
    candidate_steps: int,
) -> tuple[dict[str, object], dict[str, object]]:
    base_text = "q:\na: no\n" * old_repeats
    expanded_text = base_text + ("q2:\na: ok!\n" * new_repeats)
    base_corpus = _write_text(root / "base.txt", base_text)
    base_valid = _write_text(root / "base_valid.txt", base_text)
    expanded_corpus = _write_text(root / "expanded.txt", expanded_text)
    expanded_valid = _write_text(root / "expanded_valid.txt", expanded_text)
    base_metrics = train_transformer(
        _train_args(
            corpus=base_corpus,
            valid=base_valid,
            run=root / "base_run",
            steps=640,
        )
    )
    candidate_metrics = train_transformer(
        _train_args(
            corpus=expanded_corpus,
            valid=expanded_valid,
            run=root / "candidate_run",
            steps=candidate_steps,
            resume_checkpoint=Path(str(base_metrics["checkpoint"])),
        )
    )
    return base_metrics, candidate_metrics


def _record(record_id: str, prompt: str, target: str) -> dict[str, str]:
    return {"id": record_id, "prompt": prompt, "target": target}


if __name__ == "__main__":
    unittest.main()
