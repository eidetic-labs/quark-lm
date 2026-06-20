from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from support.frontier_metrics import metrics_with_profile_coverage
from transformer_direct_answer_frontier_reference import (
    build_direct_answer_frontier_reference,
)


class TransformerDirectAnswerFrontierReferenceTest(unittest.TestCase):
    def test_inactive_without_declared_frontier_metrics(self) -> None:
        reference = build_direct_answer_frontier_reference(
            args=SimpleNamespace(direct_answer_frontier_metrics=None),
            direct_baseline={},
            final_snapshot={},
        )

        self.assertFalse(reference["active"])
        self.assertFalse(reference["used_for_training"])
        self.assertIsNone(reference["metrics_path"])
        self.assertIsNone(reference["baseline_comparison"])
        self.assertIsNone(reference["final_comparison"])

    def test_records_baseline_and_final_frontier_comparisons(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            frontier_path = Path(temp) / "frontier_metrics.json"
            frontier_path.write_text(
                json.dumps(metrics_with_profile_coverage("frontier", {"qa": 0.5})),
                encoding="utf-8",
            )

            reference = build_direct_answer_frontier_reference(
                args=SimpleNamespace(direct_answer_frontier_metrics=frontier_path),
                direct_baseline=_snapshot(0.0),
                final_snapshot=_snapshot(0.5),
            )

        self.assertTrue(reference["active"])
        self.assertFalse(reference["used_for_training"])
        self.assertEqual(reference["metrics_path"], str(frontier_path))
        self.assertEqual(reference["frontier_run_id"], "frontier")
        self.assertFalse(reference["baseline_comparison"]["passed"])
        self.assertTrue(reference["final_comparison"]["passed"])
        self.assertFalse(reference["baseline_comparison"]["coverage_preserved"])
        self.assertTrue(reference["final_comparison"]["coverage_preserved"])


def _snapshot(coverage: float) -> dict[str, object]:
    return {
        "branch_profiles": {
            "qa": {
                "diversity": {
                    "target_unique": 2,
                    "predicted_unique": 1,
                    "target_token_coverage": coverage,
                    "dominant_predicted_rate": 1.0 - coverage,
                    "collapsed": False,
                },
                "target_rank": {
                    "top3_rate": coverage,
                    "top5_rate": coverage,
                    "avg": 2.0,
                },
            }
        },
        "branch_target_coverage_by_profile": {"qa": coverage},
        "branch_diversity_target": {
            "passed": False,
            "failed_profiles": 1,
            "passed_profiles": 0,
            "min_target_token_coverage": coverage,
            "root_cause": {"mode_counts": {"target_coverage_gap": 1}},
        },
    }


if __name__ == "__main__":
    unittest.main()
