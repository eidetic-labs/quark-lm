"""All-position answer training gives denser signal than single-position.

Demonstrates the Workstream-D lever: training on every target position per step
learns the full answer faster than sampling one random position per step.
"""

from __future__ import annotations

import random
import unittest

import support  # noqa: F401  (inserts src/ onto sys.path)
from answer_examples import AnswerExample
from support.char_model import char_model_fixture
from support.core import OptimizationConfig, ScalarOptimizer
from transformer_answer_training_steps import (
    answer_char_loss_scalars,
    train_answer_char,
    train_answer_char_all_positions,
)

TEXT = "question: q\nanswer: yes.\nno.\n"
EXAMPLE = AnswerExample(prompt="answer:", target=" yes.", source="test")
STEPS = 25
LEARNING_RATE = 0.05


def _full_answer_nll(model, tokenizer, example) -> float:
    n = len(tokenizer.encode(example.target))
    return sum(
        answer_char_loss_scalars(model, tokenizer, example, position).data
        for position in range(n)
    )


def _fresh_model():
    tokenizer, _ids, _config, model = char_model_fixture(TEXT, context_size=16, seed=5)
    model.active_optimizer = ScalarOptimizer(OptimizationConfig(optimizer="adamw"))
    return tokenizer, model


class AnswerAllPositionsTest(unittest.TestCase):
    def test_all_positions_learns_full_answer_faster(self) -> None:
        tok_all, model_all = _fresh_model()
        tok_single, model_single = _fresh_model()

        before = _full_answer_nll(model_all, tok_all, EXAMPLE)
        rng = random.Random(0)
        for _ in range(STEPS):
            train_answer_char_all_positions(model_all, tok_all, EXAMPLE, LEARNING_RATE)
            train_answer_char(model_single, tok_single, EXAMPLE, rng, LEARNING_RATE)

        after_all = _full_answer_nll(model_all, tok_all, EXAMPLE)
        after_single = _full_answer_nll(model_single, tok_single, EXAMPLE)

        # All-position training actually reduces the full-answer NLL,
        self.assertLess(after_all, before)
        # and reaches a lower full-answer NLL than single-position in equal steps.
        self.assertLess(after_all, after_single)


if __name__ == "__main__":
    unittest.main()
