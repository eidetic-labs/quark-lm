from __future__ import annotations

import unittest

from support.branch_diversity import summarize_branch_diversity_target


class TransformerBranchDiversitySummaryTest(unittest.TestCase):
    def test_branch_diversity_target_marks_collapsed_profiles(self) -> None:
        summary = summarize_branch_diversity_target(
            {
                "qa": {
                    "diversity": {
                        "target_unique": 2,
                        "predicted_unique": 1,
                        "target_token_coverage": 0.5,
                        "dominant_predicted_token": "a",
                        "dominant_predicted_rate": 1.0,
                        "collapsed": True,
                        "missing_target_tokens": [{"value": "b", "count": 1}],
                    }
                },
                "self": {
                    "diversity": {
                        "target_unique": 1,
                        "predicted_unique": 1,
                        "target_token_coverage": 1.0,
                        "dominant_predicted_token": "s",
                        "dominant_predicted_rate": 1.0,
                        "collapsed": False,
                        "missing_target_tokens": [],
                    }
                },
            }
        )

        self.assertFalse(summary["passed"])
        self.assertEqual(summary["multi_target_profiles"], 1)
        self.assertEqual(summary["passed_profiles"], 0)
        self.assertEqual(summary["failed_profiles"], 1)
        self.assertEqual(summary["max_dominant_predicted_rate"], 1.0)
        self.assertEqual(summary["min_target_token_coverage"], 0.5)
        self.assertEqual(summary["blocking_evals"][0]["name"], "qa")
        self.assertTrue(summary["blocking_evals"][0]["collapsed"])
        self.assertEqual(
            summary["root_cause"]["root_cause_hypothesis"],
            "profile_local_prediction_collapse",
        )
        self.assertEqual(
            summary["root_cause"]["mode_counts"]["prediction_collapse"],
            1,
        )

    def test_branch_diversity_target_passes_when_targets_are_covered(self) -> None:
        summary = summarize_branch_diversity_target(
            {
                "qa": {
                    "diversity": {
                        "target_unique": 2,
                        "predicted_unique": 2,
                        "target_token_coverage": 1.0,
                        "dominant_predicted_token": "a",
                        "dominant_predicted_rate": 0.5,
                        "collapsed": False,
                        "missing_target_tokens": [],
                    }
                }
            }
        )

        self.assertTrue(summary["passed"])
        self.assertEqual(summary["multi_target_profiles"], 1)
        self.assertEqual(summary["passed_profiles"], 1)
        self.assertEqual(summary["failed_profiles"], 0)
        self.assertEqual(summary["blocking_evals"], [])
        self.assertEqual(
            summary["root_cause"]["root_cause_hypothesis"],
            "no_branch_diversity_gap",
        )

    def test_branch_diversity_root_cause_detects_global_dominant_token(self) -> None:
        summary = summarize_branch_diversity_target(
            {
                "qa": {
                    "target_rank": {"avg": 14.0, "top3_rate": 0.0, "top5_rate": 0.125},
                    "diversity": {
                        "target_unique": 8,
                        "predicted_unique": 1,
                        "target_token_coverage": 0.125,
                        "dominant_predicted_token": "n",
                        "dominant_predicted_rate": 1.0,
                        "collapsed": True,
                        "missing_target_tokens": [{"value": "a", "count": 3}],
                    },
                },
                "heldout": {
                    "target_rank": {"avg": 13.5, "top3_rate": 0.0, "top5_rate": 0.125},
                    "diversity": {
                        "target_unique": 8,
                        "predicted_unique": 1,
                        "target_token_coverage": 0.125,
                        "dominant_predicted_token": "n",
                        "dominant_predicted_rate": 1.0,
                        "collapsed": True,
                        "missing_target_tokens": [{"value": "b", "count": 2}],
                    },
                },
            }
        )

        root_cause = summary["root_cause"]
        self.assertEqual(
            root_cause["root_cause_hypothesis"],
            "global_output_prior_collapse",
        )
        self.assertEqual(root_cause["severity"], "critical")
        self.assertTrue(root_cause["global_dominant_token_reuse"])
        self.assertEqual(root_cause["collapsed_profile_count"], 2)
        self.assertEqual(
            root_cause["reused_dominant_tokens"],
            [{"token": "n", "profile_count": 2, "profiles": ["heldout", "qa"]}],
        )
        self.assertIn(
            "Audit global logit priors and output-bias escape paths before adding another objective.",
            root_cause["recommendations"],
        )

    def test_branch_diversity_root_cause_detects_wrong_diverse_predictions(self) -> None:
        summary = summarize_branch_diversity_target(
            {
                "qa": {
                    "target_rank": {"avg": 3.0, "top3_rate": 0.5, "top5_rate": 1.0},
                    "diversity": {
                        "target_unique": 2,
                        "predicted_unique": 2,
                        "target_token_coverage": 0.5,
                        "dominant_predicted_token": "a",
                        "dominant_predicted_rate": 0.5,
                        "collapsed": False,
                        "missing_target_tokens": [{"value": "b", "count": 1}],
                    },
                }
            }
        )

        root_cause = summary["root_cause"]
        self.assertEqual(
            root_cause["root_cause_hypothesis"],
            "wrong_diversity_not_target_coverage",
        )
        self.assertEqual(root_cause["wrong_diverse_profile_count"], 1)
        self.assertEqual(
            root_cause["mode_counts"]["wrong_diverse_predictions"],
            1,
        )
        self.assertIn(
            "Require target-aligned coverage, not just more distinct predicted tokens.",
            root_cause["recommendations"],
        )


if __name__ == "__main__":
    unittest.main()
