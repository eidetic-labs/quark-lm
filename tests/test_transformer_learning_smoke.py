"""Regression guard: training must commit weight updates and reduce loss.

This is the sentinel that would have caught the frozen-model regression
(optimizer.update_count stuck at 0, model never leaving random init). It is
deliberately tiny and deterministic so it runs in well under a second.
"""

from __future__ import annotations

import unittest

from support.char_model import char_model_fixture, context_and_target
from support.core import OptimizationConfig, ScalarOptimizer

STEPS = 40
LEARNING_RATE = 0.05


class TransformerLearningSmokeTest(unittest.TestCase):
    def _fixture(self, gradient_accumulation_steps: int):
        tokenizer, ids, config, model = char_model_fixture("abcabc\n", seed=7)
        optimizer = ScalarOptimizer(
            OptimizationConfig(
                optimizer="adamw",
                gradient_accumulation_steps=gradient_accumulation_steps,
            )
        )
        model.active_optimizer = optimizer
        context, target = context_and_target(ids, config, tokenizer)
        return model, optimizer, context, target

    def test_training_commits_updates_and_reduces_loss(self) -> None:
        model, optimizer, context, target = self._fixture(1)
        before = model.nll(context, target)

        for _ in range(STEPS):
            model.train_step(context, target, learning_rate=LEARNING_RATE)

        # The regression guard: every step committed a real weight update.
        self.assertEqual(optimizer.update_count, STEPS)
        # And the model actually learned (overfitting one example drives this
        # far below 0.9x; the loose bound keeps the test non-flaky).
        after = model.nll(context, target)
        self.assertLess(after, 0.9 * before)

    def test_gradient_accumulation_defers_commits(self) -> None:
        model, optimizer, context, target = self._fixture(4)

        for _ in range(STEPS):
            model.train_step(context, target, learning_rate=LEARNING_RATE)

        # With accumulation=4, one weight update commits every 4 steps.
        self.assertEqual(optimizer.update_count, STEPS // 4)


if __name__ == "__main__":
    unittest.main()
