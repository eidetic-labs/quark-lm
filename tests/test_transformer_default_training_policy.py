"""Pin the default answer-train policy: committing optimizer + ungated path.

Guards the Stage 1 recovery decision so a future change can't silently revert
the default training path to a non-committing optimizer or re-enable the
candidate-and-rollback gating regime by default.
"""

from __future__ import annotations

import unittest

from support.commands import parse_args
from transformer_direct_answer_mode_flags import direct_answer_mode_flags


class TransformerDefaultTrainingPolicyTest(unittest.TestCase):
    def test_default_optimizer_commits_every_step(self) -> None:
        args = parse_args(["answer-train"])
        # AdamW (not the old sgd default) and no accumulation deferral, so every
        # step commits a real weight update.
        self.assertEqual(args.optimizer, "adamw")
        self.assertEqual(args.gradient_accumulation_steps, 1)
        self.assertEqual(args.direct_answer_learning_rate, 0.08)

    def test_default_path_is_ungated(self) -> None:
        args = parse_args(["answer-train"])
        # None of the three gate activators fire for a bare invocation:
        # (1) the default direct-answer mode is not in any baseline-floor set,
        flags = direct_answer_mode_flags(args.direct_answer_mode)
        self.assertFalse(any(flags.values()))
        # (2) no routing-repair experiment bundle, and
        self.assertIsNone(getattr(args, "experiment_bundle", None))
        # (3) no frontier-metrics guard.
        self.assertIsNone(getattr(args, "direct_answer_frontier_metrics", None))

    def test_answer_all_positions_is_opt_in(self) -> None:
        self.assertFalse(parse_args(["answer-train"]).answer_all_positions)
        self.assertTrue(
            parse_args(["answer-train", "--answer-all-positions"]).answer_all_positions
        )


if __name__ == "__main__":
    unittest.main()
