"""--combined-best wiring for the answer-train torch stage (default-OFF).

Without --combined-best the call into train_torch_answer_mixed carries NO eval
machinery (eval_every defaults 0), so production answer-train is byte-identical.
With --combined-best the stage builds a CorpusResponder once and passes it plus the
validation probe paths and eval cadence through.
"""

from __future__ import annotations

import argparse
import json
import unittest
from importlib import import_module
from pathlib import Path

import support  # noqa: F401  (inserts src/ onto sys.path)
from answer_examples import AnswerExample
from support.core import CharTokenizer, OptimizationConfig, TransformerConfig
from transformer_tiny_lm import TinyTransformerLM

import transformer_answer_torch_stage as stage

POOL = [
    AnswerExample("mia ball", " box", "fact"),
    AnswerExample("noah cup", " door", "fact"),
]
ALL_TEXT = "".join(e.prompt + e.target for e in POOL) + "\n"


def _torch_or_skip(test_case: unittest.TestCase):
    try:
        return import_module("torch")
    except ModuleNotFoundError:
        test_case.skipTest("optional PyTorch runtime is not installed")
        return None


class CombinedBestWiringTest(unittest.TestCase):
    def setUp(self) -> None:
        _torch_or_skip(self)
        self.tokenizer = CharTokenizer.train(ALL_TEXT)
        config = TransformerConfig(
            vocab_size=self.tokenizer.vocab_size, context_size=8, embedding_dim=4,
            feedforward_dim=8, seed=11,
        )
        self.model = TinyTransformerLM.init_random(config)
        self.captured = {}

        class _Captured(Exception):
            pass

        self._captured_exc = _Captured

        def fake_mixed(**kwargs):
            self.captured = kwargs
            # Abort before the trained-weights bridge -- we only assert the call kwargs.
            raise _Captured()

        self._orig = stage.train_torch_answer_mixed
        stage.train_torch_answer_mixed = fake_mixed

    def tearDown(self) -> None:
        stage.train_torch_answer_mixed = self._orig

    def _run(self, **extra) -> dict:
        args = argparse.Namespace(
            learning_rate=0.05, steps=20, seed=0, contrast_weight=1.0,
            corpus_dir=Path("corpus"), **extra,
        )
        with self.assertRaises(self._captured_exc):
            stage.train_core_answer_stage_torch(
                args, self.model, self.tokenizer, POOL,
                OptimizationConfig(optimizer="adamw", gradient_accumulation_steps=1, weight_decay=0.0),
            )
        return self.captured

    def test_without_combined_best_no_eval_machinery(self) -> None:
        captured = self._run()
        self.assertNotIn("eval_every", captured)
        self.assertNotIn("eval_responder", captured)
        self.assertNotIn("validation_probe_paths", captured)

    def test_with_combined_best_passes_responder_and_cadence(self) -> None:
        from corpus_responder import CorpusResponder

        captured = self._run(
            combined_best=True, eval_every_combined=10,
            combined_probes=["evals/qa.jsonl", "evals/unknowns.jsonl"],
            f1_floor=0.8, gen_floor=0.05,
        )
        self.assertEqual(captured["eval_every"], 10)
        self.assertIsInstance(captured["eval_responder"], CorpusResponder)
        self.assertEqual(
            [str(p) for p in captured["validation_probe_paths"]],
            ["evals/qa.jsonl", "evals/unknowns.jsonl"],
        )
        self.assertEqual(captured["f1_floor"], 0.8)
        self.assertEqual(captured["gen_floor"], 0.05)


if __name__ == "__main__":
    unittest.main()
