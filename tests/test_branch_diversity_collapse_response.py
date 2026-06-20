from __future__ import annotations

import unittest

from branch_diversity_collapse_response import (
    branch_diversity_collapse_response_delta,
    branch_diversity_collapse_response_improved,
)


class BranchDiversityCollapseResponseTests(unittest.TestCase):
    def test_detects_predicted_unique_recovery(self) -> None:
        baseline = _snapshot(predicted_unique=1, dominant_rate=1.0)
        cracked = _snapshot(predicted_unique=2, dominant_rate=0.5)

        delta = branch_diversity_collapse_response_delta(cracked, baseline)

        self.assertTrue(
            branch_diversity_collapse_response_improved(cracked, baseline)
        )
        self.assertEqual(delta["improved_profile_count"], 1)
        self.assertEqual(delta["regressed_profile_count"], 0)

    def test_rejects_rank_neutral_collapse_tie(self) -> None:
        baseline = _snapshot(predicted_unique=1, dominant_rate=1.0)
        still_collapsed = _snapshot(predicted_unique=1, dominant_rate=1.0)

        delta = branch_diversity_collapse_response_delta(
            still_collapsed,
            baseline,
        )

        self.assertFalse(
            branch_diversity_collapse_response_improved(still_collapsed, baseline)
        )
        self.assertEqual(delta["improved_profile_count"], 0)

    def test_rejects_new_collapse_regression(self) -> None:
        baseline = _snapshot(predicted_unique=2, dominant_rate=0.5)
        recollapsed = _snapshot(predicted_unique=1, dominant_rate=1.0)

        delta = branch_diversity_collapse_response_delta(recollapsed, baseline)

        self.assertFalse(
            branch_diversity_collapse_response_improved(recollapsed, baseline)
        )
        self.assertEqual(delta["regressed_profile_count"], 1)


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
