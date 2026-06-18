from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from support.commands import parse_args
from support.core import TinyTransformerLM, continuation_nll
from transformer_char_model import train_transformer


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


if __name__ == "__main__":
    unittest.main()
