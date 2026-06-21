from __future__ import annotations

import math
import unittest

import support  # noqa: F401  # puts src/ on sys.path

from epistemic_calibration import expected_calibration_error


def _candidates(*nlls):
    """Build a candidate_scores list from a sequence of target NLLs."""
    return [{"target": f"cand{i}", "target_nll": nll} for i, nll in enumerate(nlls)]


class EpistemicCalibrationTest(unittest.TestCase):
    def test_two_confident_records_one_wrong_gives_ece_half(self) -> None:
        # Both records have min target_nll == 0.0 -> confidence exp(0) == 1.0.
        # One correct, one not -> bin avg_confidence 1.0, accuracy 0.5.
        # ECE = (2/2) * |1.0 - 0.5| = 0.5.
        scored_by_set = {
            "set_a": [
                {
                    "candidate_scores": _candidates(0.0, 2.0),
                    "candidate_match": True,
                },
                {
                    "candidate_scores": _candidates(0.0, 1.5),
                    "candidate_match": False,
                },
            ]
        }
        result = expected_calibration_error(scored_by_set)
        self.assertAlmostEqual(result["ece"], 0.5)
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["n_bins"], 10)
        # Both confidences are 1.0 -> single non-empty bin (the last one).
        self.assertEqual(len(result["bins"]), 1)
        single = result["bins"][0]
        self.assertEqual(single["count"], 2)
        self.assertAlmostEqual(single["avg_confidence"], 1.0)
        self.assertAlmostEqual(single["accuracy"], 0.5)
        self.assertAlmostEqual(single["hi"], 1.0)

    def test_empty_input_gives_zero_ece(self) -> None:
        result = expected_calibration_error({})
        self.assertEqual(result["ece"], 0.0)
        self.assertEqual(result["n"], 0)
        self.assertEqual(result["bins"], [])
        self.assertEqual(result["n_bins"], 10)

    def test_min_nll_selects_lowest_and_confidence_matches(self) -> None:
        # min target_nll over candidates is 0.0 (not 3.0) -> confidence 1.0.
        scored_by_set = {
            "s": [
                {
                    "candidate_scores": _candidates(3.0, 0.0, 1.0),
                    "candidate_match": True,
                }
            ]
        }
        result = expected_calibration_error(scored_by_set)
        self.assertEqual(result["n"], 1)
        self.assertAlmostEqual(result["bins"][0]["avg_confidence"], 1.0)
        self.assertAlmostEqual(result["bins"][0]["accuracy"], 1.0)
        # Perfectly calibrated single point -> ECE 0.0.
        self.assertAlmostEqual(result["ece"], 0.0)

    def test_falls_back_to_target_nll_when_no_candidates(self) -> None:
        # No candidate_scores: confidence from record-level target_nll.
        # exp(-ln 2) == 0.5, which lands in bin [0.4, 0.5).
        scored_by_set = {
            "s": [
                {
                    "target_nll": math.log(2.0),
                    "candidate_match": None,
                    "exact_match": True,
                }
            ]
        }
        result = expected_calibration_error(scored_by_set)
        self.assertEqual(result["n"], 1)
        self.assertAlmostEqual(result["bins"][0]["avg_confidence"], 0.5)
        # candidate_match is None -> fall back to exact_match (True).
        self.assertAlmostEqual(result["bins"][0]["accuracy"], 1.0)
        self.assertAlmostEqual(result["ece"], 0.5)

    def test_skips_records_lacking_both_signals(self) -> None:
        scored_by_set = {
            "s": [
                {"candidate_match": True},  # no confidence signal -> skipped
                {
                    "candidate_scores": _candidates(0.0),
                    "candidate_match": True,
                },
            ]
        }
        result = expected_calibration_error(scored_by_set)
        self.assertEqual(result["n"], 1)

    def test_two_bins_count_weighted_average(self) -> None:
        # Bin [0.9, 1.0]: 2 records, conf 1.0, accuracy 0.5 -> gap 0.5.
        # Bin [0.4, 0.5): 1 record, conf 0.5, accuracy 1.0 -> gap 0.5.
        # ECE = (2/3)*0.5 + (1/3)*0.5 = 0.5.
        scored_by_set = {
            "a": [
                {"candidate_scores": _candidates(0.0), "candidate_match": True},
                {"candidate_scores": _candidates(0.0), "candidate_match": False},
            ],
            "b": [
                {
                    "candidate_scores": _candidates(math.log(2.0)),
                    "candidate_match": True,
                }
            ],
        }
        result = expected_calibration_error(scored_by_set)
        self.assertEqual(result["n"], 3)
        self.assertEqual(len(result["bins"]), 2)
        self.assertAlmostEqual(result["ece"], 0.5)


if __name__ == "__main__":
    unittest.main()
