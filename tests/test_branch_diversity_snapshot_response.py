from __future__ import annotations

import unittest

from support.branch_diversity import branch_diversity_snapshot_score_improved


class BranchDiversitySnapshotResponseTest(unittest.TestCase):
    def test_score_improved_detects_rank_lift_without_coverage_gain(self) -> None:
        baseline = _snapshot(
            coverage=0.25,
            target_rank=20.0,
            top3_rate=0.0,
            top5_rate=0.0,
        )
        lifted = _snapshot(
            coverage=0.25,
            target_rank=8.0,
            top3_rate=0.25,
            top5_rate=0.5,
        )

        self.assertTrue(
            branch_diversity_snapshot_score_improved(lifted, baseline)
        )


def _snapshot(
    *,
    coverage: float,
    target_rank: float,
    top3_rate: float,
    top5_rate: float,
) -> dict:
    return {
        "branch_diversity_target": {
            "passed": False,
            "passed_profiles": 0,
            "failed_profiles": 1,
            "min_target_token_coverage": coverage,
        },
        "branch_profiles": {
            "qa": {
                "diversity": {
                    "target_unique": 4,
                    "predicted_unique": 1,
                    "target_token_coverage": coverage,
                    "dominant_predicted_rate": 1.0,
                },
                "target_rank": {
                    "avg": target_rank,
                    "top3_rate": top3_rate,
                    "top5_rate": top5_rate,
                },
            }
        },
    }


if __name__ == "__main__":
    unittest.main()
