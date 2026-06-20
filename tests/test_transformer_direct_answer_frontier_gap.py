from __future__ import annotations

import unittest

from transformer_direct_answer_frontier_gap import build_frontier_gap_summary


class TransformerDirectAnswerFrontierGapTest(unittest.TestCase):
    def test_inactive_when_comparison_missing(self) -> None:
        summary = build_frontier_gap_summary(None)

        self.assertFalse(summary["active"])
        self.assertFalse(summary["available"])
        self.assertFalse(summary["used_for_training"])
        self.assertEqual(summary["repair_focus_profiles"], [])

    def test_summarizes_regressed_and_diagnostic_profiles(self) -> None:
        summary = build_frontier_gap_summary(
            {
                "available": True,
                "passed": False,
                "coverage_preserved": False,
                "stability_preserved": True,
                "score_preserved": True,
                "coverage_delta": {
                    "regressed_profiles": [
                        {"profile": "learning", "delta": -0.25},
                        {"profile": "glossary", "delta": -0.1},
                    ],
                    "improved_profile_count": 1,
                    "improved_profiles": [{"profile": "qa", "delta": 0.125}],
                    "tied_profile_count": 1,
                    "tied_profiles": ["self"],
                },
                "profile_regression_diagnostics": {
                    "diagnosis_label_counts": {
                        "target_rank_regression": 2,
                        "zero_coverage_regression": 1,
                    },
                    "profiles": [{"profile": "owner"}],
                    "worst_profile": {"profile": "learning"},
                },
            }
        )

        self.assertTrue(summary["active"])
        self.assertFalse(summary["used_for_training"])
        self.assertEqual(
            summary["coverage_regressed_profiles"],
            ["glossary", "learning"],
        )
        self.assertEqual(summary["improved_profiles"], ["qa"])
        self.assertEqual(summary["tied_profiles"], ["self"])
        self.assertEqual(summary["worst_profile"], "learning")
        self.assertEqual(
            summary["repair_focus_profiles"],
            ["glossary", "learning", "owner"],
        )
        self.assertEqual(
            summary["diagnosis_label_counts"],
            {"target_rank_regression": 2, "zero_coverage_regression": 1},
        )


if __name__ == "__main__":
    unittest.main()
