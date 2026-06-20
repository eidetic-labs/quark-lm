from __future__ import annotations

import unittest

from transformer_branch_replay_competitor_diagnostics import (
    branch_replay_competitor_movement,
    branch_replay_competitor_summary,
)


class TransformerBranchReplayCompetitorDiagnosticsTests(unittest.TestCase):
    def test_competitor_summary_reports_dominant_losing_token(self) -> None:
        tokenizer = _FakeTokenizer(["<pad>", "a", "b", "c"])
        model = _FakeModel(
            {
                (0,): [0.0, 0.2, 0.7, 0.1],
                (1,): [0.0, 0.1, 0.6, 0.3],
                (2,): [0.0, 0.8, 0.1, 0.1],
            }
        )
        branches = [
            ([0], 1, 2, "qa"),
            ([1], 3, 2, "qa"),
            ([2], 1, 1, "owner"),
        ]

        summary = branch_replay_competitor_summary(model, tokenizer, branches)

        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["target_won_count"], 1)
        self.assertEqual(summary["competitor_count"], 2)
        self.assertEqual(summary["dominant_competitor_token"], "b")
        self.assertAlmostEqual(summary["dominant_competitor_rate"], 2 / 3)
        self.assertAlmostEqual(summary["avg_losing_margin"], 0.4)
        self.assertIn("qa", summary["profiles"])
        self.assertEqual(summary["profiles"]["qa"]["dominant_competitor_token"], "b")

    def test_competitor_movement_compares_first_and_last_step(self) -> None:
        movement = branch_replay_competitor_movement(
            [
                {
                    "target_floor_competitor_summary": {
                        "target_won_rate": 0.0,
                        "dominant_competitor_rate": 0.75,
                        "avg_losing_margin": 0.5,
                    }
                },
                {
                    "target_floor_competitor_summary": {
                        "target_won_rate": 0.25,
                        "dominant_competitor_rate": 0.5,
                        "avg_losing_margin": 0.25,
                    }
                },
            ]
        )

        self.assertTrue(movement["available"])
        self.assertEqual(movement["target_won_rate_delta"], 0.25)
        self.assertEqual(movement["dominant_competitor_rate_delta"], -0.25)
        self.assertTrue(movement["target_won_rate_improved"])
        self.assertTrue(movement["dominant_competitor_rate_reduced"])
        self.assertTrue(movement["losing_margin_reduced"])


class _FakeTokenizer:
    def __init__(self, tokens: list[str]) -> None:
        self.itos = {index: token for index, token in enumerate(tokens)}


class _FakeModel:
    def __init__(self, probabilities: dict[tuple[int, ...], list[float]]) -> None:
        self._probabilities = probabilities

    def predict(self, context: list[int]) -> list[float]:
        return self._probabilities[tuple(context)]


if __name__ == "__main__":
    unittest.main()
