from __future__ import annotations

import unittest

from support.branch_diversity import (
    branch_diversity_snapshot_preserves_target_coverage,
    branch_diversity_snapshot_target_coverage_diagnostics,
)


class TransformerBranchDiversityCoverageTest(unittest.TestCase):
    def test_branch_diversity_snapshot_coverage_floor_is_profile_wise(self) -> None:
        baseline = {
            "branch_profiles": {
                "qa": {
                    "diversity": {
                        "target_unique": 4,
                        "target_token_coverage": 0.25,
                    }
                },
                "heldout": {
                    "diversity": {
                        "target_unique": 4,
                        "target_token_coverage": 0.25,
                    }
                },
                "self": {
                    "diversity": {
                        "target_unique": 1,
                        "target_token_coverage": 1.0,
                    }
                },
            }
        }
        rank_lifted_but_forgetting = {
            "branch_profiles": {
                "qa": {
                    "diversity": {
                        "target_unique": 4,
                        "target_token_coverage": 0.0,
                    },
                    "target_rank": {
                        "avg": 4.0,
                        "top3_rate": 0.5,
                        "top5_rate": 0.75,
                    },
                },
                "heldout": {
                    "diversity": {
                        "target_unique": 4,
                        "target_token_coverage": 0.5,
                    },
                    "target_rank": {
                        "avg": 4.0,
                        "top3_rate": 0.5,
                        "top5_rate": 0.75,
                    },
                },
            }
        }
        coverage_preserved = {
            "branch_profiles": {
                "qa": {
                    "diversity": {
                        "target_unique": 4,
                        "target_token_coverage": 0.25,
                    }
                },
                "heldout": {
                    "diversity": {
                        "target_unique": 4,
                        "target_token_coverage": 0.25,
                    }
                },
            }
        }

        self.assertFalse(
            branch_diversity_snapshot_preserves_target_coverage(
                rank_lifted_but_forgetting,
                baseline,
            )
        )
        self.assertTrue(
            branch_diversity_snapshot_preserves_target_coverage(
                coverage_preserved,
                baseline,
            )
        )
        diagnostics = branch_diversity_snapshot_target_coverage_diagnostics(
            rank_lifted_but_forgetting,
            baseline,
        )
        self.assertFalse(diagnostics["preserved"])
        self.assertEqual(diagnostics["violating_profile_count"], 1)
        self.assertEqual(
            diagnostics["worst_violation"],
            {
                "profile": "qa",
                "baseline_coverage": 0.25,
                "snapshot_coverage": 0.0,
                "deficit": 0.25,
            },
        )


if __name__ == "__main__":
    unittest.main()
