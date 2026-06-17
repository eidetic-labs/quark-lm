from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from support.char_model import char_model_fixture, context_and_target
from support.core import (
    OptimizationConfig,
    ScalarOptimizer,
    load_optimizer_state,
    save_optimizer_state,
)


class TransformerOptimizerStateTest(unittest.TestCase):
    def test_adamw_optimizer_accumulates_gradients_and_round_trips(self) -> None:
        tokenizer, ids, config, model = char_model_fixture("abc abc\n", seed=53)
        optimizer = ScalarOptimizer(
            OptimizationConfig(
                optimizer="adamw",
                gradient_accumulation_steps=2,
                warmup_steps=2,
                decay_steps=2,
                min_learning_rate=0.001,
            )
        )
        model.active_optimizer = optimizer
        context, target = context_and_target(ids, config, tokenizer)
        before = model.nll(context, target)

        model.train_step(context, target, learning_rate=0.02)
        self.assertEqual(optimizer.update_count, 0)
        self.assertEqual(optimizer.pending_accumulation, 1)
        model.train_step(context, target, learning_rate=0.02)
        self.assertEqual(optimizer.update_count, 1)
        self.assertEqual(optimizer.pending_accumulation, 0)
        self.assertGreater(before, model.nll(context, target))

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "optimizer.json"
            save_optimizer_state(path, optimizer)
            loaded = load_optimizer_state(path, optimizer.config)

        self.assertEqual(loaded.update_count, optimizer.update_count)
        self.assertEqual(loaded.config.optimizer, "adamw")
        self.assertEqual(len(loaded.first_moment), len(optimizer.first_moment))


if __name__ == "__main__":
    unittest.main()
