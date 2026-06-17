from __future__ import annotations

import unittest

from support.branch_diversity import branch_routing_audit_summary


class TransformerBranchRoutingAuditTest(unittest.TestCase):
    def test_branch_routing_audit_flags_output_bias_escape_risk(self) -> None:
        branch_profiles = {
            "qa": {
                "target_tokens": [{"value": "a", "count": 1}, {"value": "b", "count": 1}],
                "predicted_tokens": [{"value": "n", "count": 2}],
                "target_rank": {"avg": 20.0, "top3_rate": 0.0, "top5_rate": 0.0},
                "diversity": {
                    "target_unique": 2,
                    "predicted_unique": 1,
                    "target_token_coverage": 0.0,
                    "dominant_predicted_token": "n",
                    "dominant_predicted_rate": 1.0,
                    "collapsed": True,
                    "missing_target_tokens": [
                        {"value": "a", "count": 1},
                        {"value": "b", "count": 1},
                    ],
                },
            },
            "heldout": {
                "target_tokens": [{"value": "c", "count": 1}, {"value": "d", "count": 1}],
                "predicted_tokens": [{"value": "n", "count": 2}],
                "target_rank": {"avg": 20.0, "top3_rate": 0.0, "top5_rate": 0.0},
                "diversity": {
                    "target_unique": 2,
                    "predicted_unique": 1,
                    "target_token_coverage": 0.0,
                    "dominant_predicted_token": "n",
                    "dominant_predicted_rate": 1.0,
                    "collapsed": True,
                    "missing_target_tokens": [
                        {"value": "c", "count": 1},
                        {"value": "d", "count": 1},
                    ],
                },
            },
        }

        audit = branch_routing_audit_summary(
            branch_profiles,
            output_bias_by_token={"n": 4.0, "a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0},
            branch_logit_prior_profiles={
                "qa": {
                    "dominant_predicted_token": "n",
                    "dominant_predicted_rate": 1.0,
                    "dominant_token_bias": 4.0,
                    "dominant_token_bias_rank": 1,
                    "dominant_vs_target_decomposition": {
                        "failed_records": {
                            "count": 2,
                            "avg_bias_advantage": 4.0,
                            "avg_hidden_advantage": 0.0,
                            "avg_logit_advantage": 4.0,
                            "bias_share_of_positive_advantage": 1.0,
                            "dominant_logit_win_rate": 1.0,
                            "primary_pressure": "output_bias",
                        }
                    },
                }
            },
        )

        self.assertEqual(audit["root_cause_hypothesis"], "global_output_prior_collapse")
        self.assertEqual(audit["audit_hypothesis"], "output_bias_escape_risk")
        self.assertEqual(audit["output_bias"]["escape_risk"], "high")
        self.assertEqual(
            audit["output_bias"]["dominant_tokens"][0]["token"],
            "n",
        )
        self.assertEqual(
            audit["output_bias"]["dominant_tokens"][0]["bias_rank"],
            1,
        )
        self.assertGreater(audit["output_bias"]["dominant_bias_advantage"], 0.0)
        self.assertEqual(audit["logit_prior"]["bias_driven_profile_count"], 1)
        self.assertEqual(
            audit["logit_prior"]["profiles"][0]["primary_pressure"],
            "output_bias",
        )
        self.assertIn(
            "Compare dominant-token bias ranks against missing target-token bias ranks.",
            audit["next_checks"],
        )
        self.assertIn(
            "Trace output-bias update paths for reused dominant tokens before another repair objective.",
            audit["next_checks"],
        )

    def test_branch_routing_audit_flags_representation_and_imbalance_risk(self) -> None:
        branch_profiles = {
            "qa": {
                "target_tokens": [
                    {"value": "a", "count": 4},
                    {"value": "b", "count": 1},
                    {"value": "c", "count": 1},
                ],
                "predicted_tokens": [{"value": "x", "count": 6}],
                "target_rank": {"avg": 10.0, "top3_rate": 0.2, "top5_rate": 0.2},
                "diversity": {
                    "target_unique": 3,
                    "predicted_unique": 1,
                    "target_token_coverage": 0.0,
                    "dominant_predicted_token": "x",
                    "dominant_predicted_rate": 1.0,
                    "collapsed": True,
                    "missing_target_tokens": [
                        {"value": "a", "count": 4},
                        {"value": "b", "count": 1},
                        {"value": "c", "count": 1},
                    ],
                },
            }
        }
        representation_profiles = {
            "qa": {
                "target_unique": 3,
                "different_target_pairwise_distance": {"avg": 0.001},
                "same_target_pairwise_distance": {"avg": 0.001},
            }
        }

        audit = branch_routing_audit_summary(
            branch_profiles,
            representation_profiles,
            output_bias_by_token={"x": 0.0, "a": 0.0, "b": 0.0, "c": 0.0},
        )

        self.assertEqual(
            audit["audit_hypothesis"],
            "representation_separation_risk",
        )
        self.assertEqual(audit["representation"]["low_separation_profile_count"], 1)
        self.assertEqual(
            audit["target_imbalance"]["high_imbalance_profiles"][0]["profile"],
            "qa",
        )
        self.assertEqual(audit["target_imbalance"]["rare_target_token_count"], 2)
        self.assertIn(
            "Balance candidate construction by profile and target token before another objective screen.",
            audit["next_checks"],
        )


if __name__ == "__main__":
    unittest.main()
