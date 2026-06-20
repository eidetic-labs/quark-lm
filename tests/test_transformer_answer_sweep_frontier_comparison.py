from __future__ import annotations

import unittest
from pathlib import Path

from support.frontier_metrics import (
    metrics_with_profile,
    metrics_with_profile_coverage,
)
from transformer_answer_sweep_report import trial_report_from_metrics


class TransformerAnswerSweepFrontierComparisonTest(unittest.TestCase):
    def test_trial_report_compares_branch_evidence_to_frontier(self) -> None:
        report = trial_report_from_metrics(
            trial_id="trial-01",
            run_path=Path("runs/trial-01"),
            config={"embedding_dim": 8},
            metrics=metrics_with_profile_coverage("trial", {"qa": 0.125}),
            frontier_metrics=metrics_with_profile_coverage(
                "frontier",
                {"qa": 0.25},
            ),
        )

        comparison = report["frontier_comparison"]
        self.assertFalse(comparison["passed"])
        self.assertEqual(comparison["frontier_run_id"], "frontier")
        self.assertFalse(comparison["coverage_preserved"])
        self.assertFalse(comparison["stability_preserved"])
        self.assertEqual(
            comparison["coverage_diagnostics"]["worst_violation"]["profile"],
            "qa",
        )
        self.assertEqual(
            comparison["stability_diagnostics"]["worst_violation"]["reason"],
            "dominant_rate_regression",
        )
        self.assertEqual(
            comparison["profile_regression_diagnostics"]["worst_profile"]["profile"],
            "qa",
        )

    def test_trial_report_fails_when_frontier_stability_regresses(self) -> None:
        report = trial_report_from_metrics(
            trial_id="trial-01",
            run_path=Path("runs/trial-01"),
            config={"embedding_dim": 8},
            metrics=metrics_with_profile(
                "trial",
                coverage=0.5,
                dominant_rate=0.75,
                top3_rate=0.75,
                top5_rate=0.75,
                avg_rank=1.5,
            ),
            frontier_metrics=metrics_with_profile(
                "frontier",
                coverage=0.5,
                dominant_rate=0.5,
                top3_rate=0.5,
                top5_rate=0.5,
                avg_rank=2.0,
            ),
        )

        comparison = report["frontier_comparison"]

        self.assertFalse(comparison["passed"])
        self.assertTrue(comparison["coverage_preserved"])
        self.assertTrue(comparison["score_preserved"])
        self.assertFalse(comparison["stability_preserved"])
        self.assertEqual(
            comparison["stability_diagnostics"]["dominant_rate_regression_count"],
            1,
        )

    def test_trial_report_summarizes_direct_answer_frontier_reference(self) -> None:
        metrics = metrics_with_profile_coverage("trial", {"qa": 0.25})
        metrics["direct_answer"]["direct_answer_frontier_reference"] = {
            "active": True,
            "used_for_training": False,
            "metrics_path": "runs/frontier/transformer_answer_metrics.json",
            "frontier_run_id": "frontier",
            "baseline_comparison": {
                "available": True,
                "passed": False,
                "coverage_preserved": False,
                "stability_preserved": False,
                "score_preserved": False,
                "stability_diagnostics": {
                    "violating_profile_count": 1,
                    "worst_violation": {"profile": "qa"},
                },
            },
            "final_comparison": {
                "available": True,
                "passed": True,
                "coverage_preserved": True,
                "stability_preserved": True,
                "score_preserved": True,
                "stability_diagnostics": {
                    "violating_profile_count": 0,
                    "worst_violation": None,
                },
            },
        }

        report = trial_report_from_metrics(
            trial_id="trial-01",
            run_path=Path("runs/trial-01"),
            config={"embedding_dim": 8},
            metrics=metrics,
        )

        reference = report["direct_answer_frontier_reference"]
        self.assertTrue(reference["active"])
        self.assertFalse(reference["used_for_training"])
        self.assertFalse(reference["baseline_comparison"]["passed"])
        self.assertTrue(reference["final_comparison"]["passed"])
        self.assertEqual(
            reference["baseline_comparison"]["stability_violating_profile_count"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
