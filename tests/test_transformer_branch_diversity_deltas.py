from __future__ import annotations

import unittest

from support.branch_diversity import (
    branch_diversity_snapshot_collapsed_profile_names,
    branch_diversity_snapshot_profile_diversity_delta,
    branch_diversity_snapshot_target_coverage_delta,
)


class TransformerBranchDiversityDeltasTest(unittest.TestCase):
    def test_branch_diversity_target_coverage_delta_records_profile_gains(
        self,
    ) -> None:
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
                        "target_unique": 3,
                        "target_token_coverage": 1.0 / 3.0,
                    }
                },
            }
        }
        snapshot = {
            "branch_profiles": {
                "qa": {
                    "diversity": {
                        "target_unique": 4,
                        "target_token_coverage": 0.5,
                    }
                },
                "heldout": {
                    "diversity": {
                        "target_unique": 3,
                        "target_token_coverage": 1.0 / 3.0,
                    }
                },
            }
        }

        delta = branch_diversity_snapshot_target_coverage_delta(snapshot, baseline)

        self.assertEqual(delta["improved_profile_count"], 1)
        self.assertEqual(delta["regressed_profile_count"], 0)
        self.assertEqual(delta["tied_profile_count"], 1)
        self.assertAlmostEqual(delta["min_delta"], 1.0 / 12.0)
        self.assertAlmostEqual(delta["average_delta"], 0.125)

    def test_branch_diversity_collapsed_profile_delta_tracks_targeted_gain(
        self,
    ) -> None:
        baseline = {
            "branch_diversity_target": {
                "blocking_evals": [
                    {"name": "owner", "collapsed": True},
                    {"name": "qa", "collapsed": False},
                ]
            },
            "branch_profiles": {
                "owner": {
                    "diversity": {
                        "predicted_unique": 1,
                        "target_token_coverage": 0.125,
                        "dominant_predicted_rate": 1.0,
                    }
                },
                "qa": {
                    "diversity": {
                        "predicted_unique": 2,
                        "target_token_coverage": 0.25,
                        "dominant_predicted_rate": 0.5,
                    }
                },
            },
        }
        snapshot = {
            "branch_profiles": {
                "owner": {
                    "diversity": {
                        "predicted_unique": 2,
                        "target_token_coverage": 0.125,
                        "dominant_predicted_rate": 0.75,
                    }
                },
                "qa": {
                    "diversity": {
                        "predicted_unique": 2,
                        "target_token_coverage": 0.25,
                        "dominant_predicted_rate": 0.5,
                    }
                },
            }
        }

        collapsed_profiles = branch_diversity_snapshot_collapsed_profile_names(
            baseline
        )
        delta = branch_diversity_snapshot_profile_diversity_delta(
            snapshot,
            baseline,
            collapsed_profiles,
        )

        self.assertEqual(collapsed_profiles, ["owner"])
        self.assertEqual(delta["improved_profile_count"], 1)
        self.assertEqual(delta["regressed_profile_count"], 0)
        self.assertEqual(
            delta["improved_profiles"][0]["predicted_unique_delta"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
