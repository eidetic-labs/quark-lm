from __future__ import annotations

import unittest

from support.branch_diversity import (
    branch_diversity_snapshot_preserves_profile_stability,
    branch_diversity_snapshot_stability_diagnostics,
)


class TransformerBranchDiversityStabilityTest(unittest.TestCase):
    def test_preserves_stability_when_profile_metrics_do_not_regress(self) -> None:
        baseline = _snapshot(predicted_unique=2, dominant_rate=0.5)
        snapshot = _snapshot(predicted_unique=2, dominant_rate=0.5)

        self.assertTrue(
            branch_diversity_snapshot_preserves_profile_stability(
                snapshot,
                baseline,
            )
        )
        diagnostics = branch_diversity_snapshot_stability_diagnostics(
            snapshot,
            baseline,
        )
        self.assertTrue(diagnostics["preserved"])
        self.assertEqual(diagnostics["violating_profile_count"], 0)

    def test_rejects_new_profile_collapse(self) -> None:
        baseline = _snapshot(predicted_unique=2, dominant_rate=0.5)
        snapshot = _snapshot(predicted_unique=1, dominant_rate=1.0)

        diagnostics = branch_diversity_snapshot_stability_diagnostics(
            snapshot,
            baseline,
        )

        self.assertFalse(diagnostics["preserved"])
        self.assertEqual(diagnostics["violating_profile_count"], 1)
        self.assertEqual(diagnostics["newly_collapsed_profile_count"], 1)
        self.assertEqual(
            diagnostics["worst_violation"]["reason"],
            "newly_collapsed",
        )

    def test_rejects_dominant_rate_regression_without_collapse(self) -> None:
        baseline = _snapshot(predicted_unique=2, dominant_rate=0.5)
        snapshot = _snapshot(predicted_unique=2, dominant_rate=0.75)

        diagnostics = branch_diversity_snapshot_stability_diagnostics(
            snapshot,
            baseline,
        )

        self.assertFalse(diagnostics["preserved"])
        self.assertEqual(diagnostics["dominant_rate_regression_count"], 1)
        self.assertEqual(
            diagnostics["worst_violation"]["reason"],
            "dominant_rate_regression",
        )


def _snapshot(predicted_unique: int, dominant_rate: float) -> dict:
    return {
        "branch_profiles": {
            "qa": {
                "diversity": {
                    "target_unique": 2,
                    "predicted_unique": predicted_unique,
                    "dominant_predicted_rate": dominant_rate,
                }
            }
        }
    }


if __name__ == "__main__":
    unittest.main()
