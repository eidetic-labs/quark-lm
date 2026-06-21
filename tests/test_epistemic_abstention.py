from __future__ import annotations

import unittest

import support  # noqa: F401  (puts src/ on sys.path)

from epistemic_abstention import abstention_metrics


def _build_scored_by_set() -> dict:
    """A 2x2 synthetic eval: one record per confusion cell.

    "unknowns" set: out-of-corpus queries (gold target " unknown.")
        - predicted " unknown."   -> true positive
        - predicted " in the bag." -> false negative (confabulated)
    "qa" set: in-corpus queries (real gold answers)
        - predicted " unknown."   -> false positive (wrongly refused)
        - predicted correct answer -> true negative (correctly answered)
    """
    return {
        "unknowns": [
            {
                "id": "u-tp",
                "target": " unknown.",
                "predicted_candidate": " unknown.",
            },
            {
                "id": "u-fn",
                "target": " unknown.",
                "predicted_candidate": " in the bag.",
            },
        ],
        "qa": [
            {
                "id": "q-fp",
                "target": " in the bag.",
                "predicted_candidate": " unknown.",
            },
            {
                "id": "q-tn",
                "target": " in the bag.",
                "predicted_candidate": " in the bag.",
            },
        ],
    }


class EpistemicAbstentionTest(unittest.TestCase):
    def test_balanced_confusion_matrix_and_rates(self) -> None:
        metrics = abstention_metrics(_build_scored_by_set())

        counts = metrics["counts"]
        self.assertEqual(counts["tp"], 1)
        self.assertEqual(counts["fp"], 1)
        self.assertEqual(counts["fn"], 1)
        self.assertEqual(counts["tn"], 1)
        self.assertEqual(counts["total"], 4)
        self.assertEqual(counts["should_abstain"], 2)
        self.assertEqual(counts["model_abstained"], 2)

        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)

    def test_per_set_breakdown(self) -> None:
        metrics = abstention_metrics(_build_scored_by_set())
        per_set = metrics["per_set"]
        self.assertEqual(set(per_set), {"unknowns", "qa"})

        unknowns = per_set["unknowns"]
        self.assertEqual(unknowns["tp"], 1)
        self.assertEqual(unknowns["fp"], 0)
        self.assertEqual(unknowns["fn"], 1)
        self.assertEqual(unknowns["tn"], 0)
        self.assertEqual(unknowns["total"], 2)
        self.assertEqual(unknowns["should_abstain"], 2)
        self.assertEqual(unknowns["model_abstained"], 1)
        # No fp in this set: precision = 1/(1+0) = 1.0; recall = 1/(1+1) = 0.5.
        self.assertEqual(unknowns["precision"], 1.0)
        self.assertEqual(unknowns["recall"], 0.5)
        self.assertAlmostEqual(unknowns["f1"], 2 / 3)

        qa = per_set["qa"]
        self.assertEqual(qa["tp"], 0)
        self.assertEqual(qa["fp"], 1)
        self.assertEqual(qa["fn"], 0)
        self.assertEqual(qa["tn"], 1)
        self.assertEqual(qa["total"], 2)
        self.assertEqual(qa["should_abstain"], 0)
        self.assertEqual(qa["model_abstained"], 1)
        # No tp in this set: precision = 0/(0+1) = 0.0; recall undefined (no
        # should-abstain), so f1 is None.
        self.assertEqual(qa["precision"], 0.0)
        self.assertIsNone(qa["recall"])
        self.assertIsNone(qa["f1"])

    def test_completion_fallback_when_no_candidate(self) -> None:
        scored = {
            "fallback": [
                # predicted_candidate None -> use completion; "unknown." (no
                # leading space) counts as abstaining.
                {"target": " unknown.", "predicted_candidate": None,
                 "completion": "unknown."},
                # completion is not the abstain token -> not abstained -> fn.
                {"target": " unknown.", "predicted_candidate": None,
                 "completion": "the river."},
            ]
        }
        counts = abstention_metrics(scored)["counts"]
        self.assertEqual(counts["tp"], 1)
        self.assertEqual(counts["fn"], 1)
        self.assertEqual(counts["fp"], 0)
        self.assertEqual(counts["tn"], 0)

    def test_empty_input_does_not_crash(self) -> None:
        metrics = abstention_metrics({})
        self.assertEqual(metrics["counts"], {
            "tp": 0, "fp": 0, "fn": 0, "tn": 0,
            "total": 0, "should_abstain": 0, "model_abstained": 0,
        })
        self.assertIsNone(metrics["precision"])
        self.assertIsNone(metrics["recall"])
        self.assertIsNone(metrics["f1"])
        self.assertEqual(metrics["per_set"], {})


if __name__ == "__main__":
    unittest.main()
