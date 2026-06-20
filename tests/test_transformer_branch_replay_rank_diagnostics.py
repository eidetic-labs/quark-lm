from __future__ import annotations

import unittest

from support.branch_training import branch_training_fixture
from transformer_branch_replay_rank_diagnostics import (
    branch_replay_rank_movement,
    branch_replay_rank_summary,
)


class TransformerBranchReplayRankDiagnosticsTests(unittest.TestCase):
    def test_rank_summary_reports_profile_rank_rates(self) -> None:
        fixture = branch_training_fixture(seed=91)
        target = fixture.tokenizer.stoi[fixture.near.target[1]]
        context = [fixture.tokenizer.pad_id] * fixture.model.config.context_size
        branches = [(context, target, target, "qa")]

        summary = branch_replay_rank_summary(fixture.model, branches)

        self.assertEqual(summary["count"], 1)
        self.assertEqual(summary["target_count"], 1)
        self.assertIn("qa", summary["profiles"])
        self.assertGreaterEqual(summary["avg_target_rank"], 1.0)
        self.assertLessEqual(summary["top1_rate"], 1.0)

    def test_rank_movement_compares_first_and_last_step(self) -> None:
        movement = branch_replay_rank_movement(
            [
                {
                    "target_floor_rank_summary": {
                        "avg_target_rank": 8.0,
                        "top1_rate": 0.0,
                    }
                },
                {
                    "target_floor_rank_summary": {
                        "avg_target_rank": 5.0,
                        "top1_rate": 0.25,
                    }
                },
            ]
        )

        self.assertTrue(movement["available"])
        self.assertEqual(movement["avg_target_rank_delta"], -3.0)
        self.assertEqual(movement["top1_rate_delta"], 0.25)
        self.assertTrue(movement["rank_improved"])
        self.assertTrue(movement["top1_improved"])


if __name__ == "__main__":
    unittest.main()
