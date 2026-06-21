from __future__ import annotations

import unittest

import support  # noqa: F401  (inserts src/ onto sys.path)
from epistemic_report import epistemic_report


class _StubResponder:
    """Minimal closed-world oracle: abstains on 'bob', answers otherwise."""

    def answer_prompt(self, prompt: str) -> str:
        return " unknown." if "bob" in prompt else " kitchen."


def _data():
    evals = {
        "qa": {"avg_target_nll": 2.0, "count": 2},
        "unknowns": {"avg_target_nll": 4.0, "count": 2},
    }
    scored_by_set = {
        "qa": [
            {
                "prompt": "question: where is alice's bag?\nanswer:",
                "target": " kitchen.",
                "predicted_candidate": " kitchen.",
                "candidate_match": True,
                "candidate_scores": [{"target": " kitchen.", "target_nll": 0.2}],
            },
            {
                "prompt": "question: where is alice's bag?\nanswer:",
                "target": " kitchen.",
                "predicted_candidate": " unknown.",
                "candidate_match": False,
                "candidate_scores": [{"target": " unknown.", "target_nll": 1.0}],
            },
        ],
        "unknowns": [
            {
                "prompt": "question: where is bob's hat?\nanswer:",
                "target": " unknown.",
                "predicted_candidate": " unknown.",
                "candidate_match": True,
                "candidate_scores": [{"target": " unknown.", "target_nll": 0.1}],
            },
        ],
    }
    return evals, scored_by_set


class EpistemicReportTest(unittest.TestCase):
    def test_report_combines_all_metrics_and_headline(self) -> None:
        evals, scored = _data()
        report = epistemic_report(evals, scored, vocab_size=36, responder=_StubResponder())

        for key in ("nll_vs_random", "abstention", "calibration", "oracle", "headline"):
            self.assertIn(key, report)

        head = report["headline"]
        # qa learned (2.0 < ln36 ~= 3.584); unknowns not (4.0 > floor).
        self.assertFalse(head["learned_all"])
        self.assertTrue(head["learned_any"])
        self.assertIsNotNone(head["mean_nll_reduction"])

        # tp=1 (unknowns abstained), fp=1 (qa rec2 wrongly abstained), fn=0, tn=1.
        counts = report["abstention"]["counts"]
        self.assertEqual((counts["tp"], counts["fp"], counts["fn"]), (1, 1, 0))
        self.assertAlmostEqual(report["abstention"]["precision"], 0.5)
        self.assertAlmostEqual(report["abstention"]["recall"], 1.0)

        self.assertIsInstance(report["calibration"]["ece"], float)
        # Stub oracle answers every probe's target correctly here.
        self.assertAlmostEqual(report["oracle"]["overall"]["oracle_exact_rate"], 1.0)
        self.assertAlmostEqual(head["oracle_exact_rate"], 1.0)

    def test_report_without_responder_omits_oracle(self) -> None:
        evals, scored = _data()
        report = epistemic_report(evals, scored, vocab_size=36)
        self.assertNotIn("oracle", report)
        self.assertIsNone(report["headline"]["oracle_exact_rate"])


if __name__ == "__main__":
    unittest.main()
