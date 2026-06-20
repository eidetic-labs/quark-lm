from __future__ import annotations

import unittest

from transformer_branch_frontier_profile_diagnostics import (
    branch_frontier_profile_regression_diagnostics,
)


class BranchFrontierProfileDiagnosticsTest(unittest.TestCase):
    def test_regressed_profile_records_routing_and_representation_causes(self) -> None:
        diagnostics = branch_frontier_profile_regression_diagnostics(
            snapshot=_snapshot(
                coverage=0.0,
                predicted_unique=1,
                dominant_rate=1.0,
                collapsed=True,
                avg_rank=12.0,
                top3_rate=0.0,
                top5_rate=0.0,
                margin_min=-0.2,
                hidden_advantage=0.8,
            ),
            frontier_snapshot=_snapshot(
                coverage=0.5,
                predicted_unique=2,
                dominant_rate=0.5,
                collapsed=False,
                avg_rank=4.0,
                top3_rate=0.5,
                top5_rate=0.75,
                margin_min=0.1,
                hidden_advantage=0.2,
            ),
        )

        profile = diagnostics["worst_profile"]

        self.assertEqual(diagnostics["regressed_profile_count"], 1)
        self.assertEqual(diagnostics["zero_coverage_regression_count"], 1)
        self.assertEqual(diagnostics["prediction_collapse_regression_count"], 1)
        self.assertEqual(diagnostics["target_rank_regression_count"], 1)
        self.assertEqual(diagnostics["target_topk_regression_count"], 1)
        self.assertEqual(diagnostics["hidden_pressure_regression_count"], 1)
        self.assertEqual(diagnostics["representation_margin_regression_count"], 1)
        self.assertEqual(profile["profile"], "qa")
        self.assertEqual(profile["coverage"]["delta"], -0.5)
        self.assertEqual(profile["dominant_prediction"]["predicted_unique_delta"], -1)
        self.assertEqual(profile["target_rank"]["avg_delta"], 8.0)
        self.assertAlmostEqual(
            profile["representation"]["target_centroid_margin_min_delta"],
            -0.3,
        )
        self.assertAlmostEqual(
            profile["logit_prior"]["avg_hidden_advantage_delta"],
            0.6,
        )
        self.assertIn(
            "hidden_projection_pressure_regression",
            profile["diagnosis_labels"],
        )


def _snapshot(
    *,
    coverage: float,
    predicted_unique: int,
    dominant_rate: float,
    collapsed: bool,
    avg_rank: float,
    top3_rate: float,
    top5_rate: float,
    margin_min: float,
    hidden_advantage: float,
) -> dict[str, object]:
    return {
        "branch_profiles": {
            "qa": {
                "diversity": {
                    "target_unique": 2,
                    "target_token_coverage": coverage,
                    "predicted_unique": predicted_unique,
                    "dominant_predicted_token": "q",
                    "dominant_predicted_rate": dominant_rate,
                    "collapsed": collapsed,
                },
                "target_rank": {
                    "avg": avg_rank,
                    "top3_rate": top3_rate,
                    "top5_rate": top5_rate,
                },
            }
        },
        "branch_representation_profiles": {
            "qa": {
                "target_centroid_margin": {
                    "min": margin_min,
                    "poorly_separated_rate": 1.0 if margin_min <= 0.0 else 0.0,
                },
                "different_target_pairwise_distance": {"avg": 0.2},
            }
        },
        "branch_logit_prior_profiles": {
            "qa": {
                "dominant_vs_target_decomposition": {
                    "failed_records": {
                        "count": 2,
                        "avg_hidden_advantage": hidden_advantage,
                        "dominant_logit_win_rate": 1.0,
                        "primary_pressure": "hidden_projection",
                    }
                }
            }
        },
    }


if __name__ == "__main__":
    unittest.main()
