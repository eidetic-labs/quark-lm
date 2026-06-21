"""Fast guard for the answer-train torch backend stage (--backend pytorch).

Asserts the all-position pair decomposition is correct and that the torch core
stage returns a trained TinyTransformerLM whose loss on a training pair drops
below the from-scratch random init. Skip-safe when torch is not installed.
"""

from __future__ import annotations

import argparse
import unittest
from importlib import import_module

import support  # noqa: F401  (inserts src/ onto sys.path)
from answer_examples import AnswerExample
from support.core import CharTokenizer, OptimizationConfig, TransformerConfig
from transformer_answer_torch_stage import (
    build_torch_answer_training_pairs,
    train_core_answer_stage_torch,
)
from transformer_tiny_lm import TinyTransformerLM

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


class AnswerTorchStageTest(unittest.TestCase):
    def test_pairs_decompose_all_target_positions(self) -> None:
        tokenizer = CharTokenizer.train(ALL_TEXT)
        example = AnswerExample("ab", "cd", "fact")
        pairs = build_torch_answer_training_pairs([example], tokenizer, 8, tokenizer.pad_id)
        target_ids = tokenizer.encode("cd")
        self.assertEqual(len(pairs), len(target_ids))
        self.assertEqual([target for _context, target in pairs], target_ids)
        # the second context must include the first emitted target char (teacher forcing)
        self.assertIn(target_ids[0], pairs[1][0])

    def test_torch_stage_returns_trained_model(self) -> None:
        _torch_or_skip(self)
        tokenizer = CharTokenizer.train(ALL_TEXT)
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size, context_size=8, embedding_dim=4, feedforward_dim=8, seed=11
        )
        model = TinyTransformerLM.init_random(config)
        pairs = build_torch_answer_training_pairs(POOL, tokenizer, 8, tokenizer.pad_id)
        sample_context, sample_target = pairs[0]
        before = model.nll(sample_context, sample_target)

        args = argparse.Namespace(learning_rate=0.05, steps=40)
        trained = train_core_answer_stage_torch(
            args, model, tokenizer, POOL,
            OptimizationConfig(optimizer="adamw", gradient_accumulation_steps=1, weight_decay=0.0),
        )

        self.assertIsInstance(trained, TinyTransformerLM)
        self.assertLess(trained.nll(sample_context, sample_target), before)


if __name__ == "__main__":
    unittest.main()
