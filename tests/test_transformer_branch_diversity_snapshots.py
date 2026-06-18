from __future__ import annotations

import unittest

from support.branch_diversity import (
    branch_diversity_snapshot_score,
)


class TransformerBranchDiversitySnapshotsTest(unittest.TestCase):
    def test_branch_diversity_snapshot_score_prefers_more_prediction_diversity(self) -> None:
        collapsed = {
            "branch_diversity_target": {
                "passed": False,
                "passed_profiles": 0,
                "failed_profiles": 1,
                "min_target_token_coverage": 0.0,
            },
            "branch_profiles": {
                "qa": {
                    "diversity": {
                        "target_unique": 4,
                        "predicted_unique": 1,
                        "target_token_coverage": 0.0,
                        "dominant_predicted_rate": 1.0,
                    }
                }
            },
        }
        cracked = {
            "branch_diversity_target": {
                "passed": False,
                "passed_profiles": 0,
                "failed_profiles": 1,
                "min_target_token_coverage": 0.0,
            },
            "branch_profiles": {
                "qa": {
                    "diversity": {
                        "target_unique": 4,
                        "predicted_unique": 2,
                        "target_token_coverage": 0.0,
                        "dominant_predicted_rate": 0.75,
                    }
                }
            },
        }

        self.assertGreater(
            branch_diversity_snapshot_score(cracked),
            branch_diversity_snapshot_score(collapsed),
        )

    def test_branch_diversity_snapshot_score_uses_target_rank_tiebreaker(self) -> None:
        buried = {
            "branch_diversity_target": {
                "passed": False,
                "passed_profiles": 0,
                "failed_profiles": 1,
                "min_target_token_coverage": 0.0,
            },
            "branch_profiles": {
                "qa": {
                    "diversity": {
                        "target_unique": 4,
                        "predicted_unique": 1,
                        "target_token_coverage": 0.0,
                        "dominant_predicted_rate": 1.0,
                    },
                    "target_rank": {
                        "avg": 20.0,
                        "top3_rate": 0.0,
                        "top5_rate": 0.0,
                    },
                }
            },
        }
        lifted = {
            "branch_diversity_target": {
                "passed": False,
                "passed_profiles": 0,
                "failed_profiles": 1,
                "min_target_token_coverage": 0.0,
            },
            "branch_profiles": {
                "qa": {
                    "diversity": {
                        "target_unique": 4,
                        "predicted_unique": 1,
                        "target_token_coverage": 0.0,
                        "dominant_predicted_rate": 1.0,
                    },
                    "target_rank": {
                        "avg": 8.0,
                        "top3_rate": 0.25,
                        "top5_rate": 0.5,
                    },
                }
            },
        }

        self.assertGreater(
            branch_diversity_snapshot_score(lifted),
            branch_diversity_snapshot_score(buried),
        )

    def test_branch_diversity_snapshot_score_preserves_cracked_collapse(self) -> None:
        cracked = {
            "branch_diversity_target": {
                "passed": False,
                "passed_profiles": 0,
                "failed_profiles": 1,
                "min_target_token_coverage": 0.0,
            },
            "branch_profiles": {
                "qa": {
                    "diversity": {
                        "target_unique": 4,
                        "predicted_unique": 2,
                        "target_token_coverage": 0.1,
                        "dominant_predicted_rate": 0.75,
                        "collapsed": False,
                    },
                    "target_rank": {
                        "avg": 12.0,
                        "top3_rate": 0.25,
                        "top5_rate": 0.25,
                    },
                }
            },
        }
        recollapsed_rank_lifted = {
            "branch_diversity_target": {
                "passed": False,
                "passed_profiles": 0,
                "failed_profiles": 1,
                "min_target_token_coverage": 0.0,
            },
            "branch_profiles": {
                "qa": {
                    "diversity": {
                        "target_unique": 4,
                        "predicted_unique": 1,
                        "target_token_coverage": 0.2,
                        "dominant_predicted_rate": 1.0,
                        "collapsed": True,
                    },
                    "target_rank": {
                        "avg": 8.0,
                        "top3_rate": 0.5,
                        "top5_rate": 0.5,
                    },
                }
            },
        }

        self.assertGreater(
            branch_diversity_snapshot_score(cracked),
            branch_diversity_snapshot_score(recollapsed_rank_lifted),
        )

    def test_branch_diversity_snapshot_score_prefers_rank_over_wrong_diversity(self) -> None:
        wrong_diverse = {
            "branch_diversity_target": {
                "passed": False,
                "passed_profiles": 0,
                "failed_profiles": 1,
                "min_target_token_coverage": 0.0,
            },
            "branch_profiles": {
                "qa": {
                    "diversity": {
                        "target_unique": 4,
                        "predicted_unique": 2,
                        "target_token_coverage": 0.0,
                        "dominant_predicted_rate": 0.75,
                    },
                    "target_rank": {
                        "avg": 14.0,
                        "top3_rate": 0.0,
                        "top5_rate": 0.25,
                    },
                }
            },
        }
        rank_lifted = {
            "branch_diversity_target": {
                "passed": False,
                "passed_profiles": 0,
                "failed_profiles": 1,
                "min_target_token_coverage": 0.0,
            },
            "branch_profiles": {
                "qa": {
                    "diversity": {
                        "target_unique": 4,
                        "predicted_unique": 1,
                        "target_token_coverage": 0.0,
                        "dominant_predicted_rate": 1.0,
                    },
                    "target_rank": {
                        "avg": 12.0,
                        "top3_rate": 0.25,
                        "top5_rate": 0.25,
                    },
                }
            },
        }

        self.assertGreater(
            branch_diversity_snapshot_score(rank_lifted),
            branch_diversity_snapshot_score(wrong_diverse),
        )


if __name__ == "__main__":
    unittest.main()
