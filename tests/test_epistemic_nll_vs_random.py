from __future__ import annotations

import math
import unittest

import support  # noqa: F401  # puts src/ on sys.path

from epistemic_nll_vs_random import nll_vs_random


class EpistemicNllVsRandomTest(unittest.TestCase):
    def test_scores_learned_and_unlearned_sets_against_random_floor(self) -> None:
        vocab_size = 36
        evals = {
            "qa": {"avg_target_nll": 2.2, "count": 10},
            "unknowns": {"avg_target_nll": 4.0, "count": 5},
            "no_nll": {"count": 7},  # missing avg_target_nll -> skipped
        }

        result = nll_vs_random(evals, vocab_size)

        # random_floor == ln(36) ~= 3.584
        self.assertAlmostEqual(result["random_floor"], math.log(36), places=6)
        self.assertAlmostEqual(result["random_floor"], 3.584, places=3)
        self.assertEqual(result["vocab_size"], vocab_size)

        per_set = result["per_set"]
        # qa learned with positive reduction.
        self.assertTrue(per_set["qa"]["learned"])
        self.assertGreater(per_set["qa"]["reduction"], 0.0)
        self.assertAlmostEqual(per_set["qa"]["avg_target_nll"], 2.2, places=6)
        # unknowns sits above the floor -> not learned.
        self.assertFalse(per_set["unknowns"]["learned"])
        # the set lacking avg_target_nll is absent from per_set.
        self.assertNotIn("no_nll", per_set)

        overall = result["overall"]
        self.assertEqual(overall["sets_scored"], 2)
        self.assertFalse(overall["learned_all"])
        self.assertTrue(overall["learned_any"])

    def test_empty_evals_does_not_crash(self) -> None:
        result = nll_vs_random({}, 36)
        self.assertEqual(result["per_set"], {})
        self.assertEqual(result["overall"]["sets_scored"], 0)
        self.assertFalse(result["overall"]["learned_all"])
        self.assertFalse(result["overall"]["learned_any"])

    def test_degenerate_vocab_has_zero_floor(self) -> None:
        result = nll_vs_random({"qa": {"avg_target_nll": 0.5, "count": 3}}, 1)
        self.assertEqual(result["random_floor"], 0.0)
        # reduction is 0.0 when the floor is zero; not learned (0.5 < 0.0 False).
        self.assertEqual(result["per_set"]["qa"]["reduction"], 0.0)
        self.assertFalse(result["per_set"]["qa"]["learned"])


if __name__ == "__main__":
    unittest.main()
