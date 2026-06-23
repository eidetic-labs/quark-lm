"""CombinedBestCheckpointTracker: gated-harmonic does-both selection.

The combined tracker selects the checkpoint that BOTH abstains (F1) AND generates
concrete answers, using a gated harmonic mean so neither axis can be traded away.
An over-abstainer (F1 high, concrete-gen 0) must score strictly below a balanced
step; a plain mean would reward abstain-everywhere. Below either floor scores 0.
"""

from __future__ import annotations

import unittest
from importlib import import_module

import support  # noqa: F401  (puts src/ on sys.path)

from transformer_best_checkpoint import CombinedBestCheckpointTracker


def _torch_or_skip(test_case: unittest.TestCase):
    try:
        return import_module("torch")
    except ModuleNotFoundError:
        test_case.skipTest("optional PyTorch runtime is not installed")
        return None


class CombinedBestCheckpointTrackerTest(unittest.TestCase):
    def test_seesaw_selects_balanced_step(self) -> None:
        torch = _torch_or_skip(self)
        params = [torch.tensor([0.0], dtype=torch.float64)]
        tracker = CombinedBestCheckpointTracker(f1_floor=0.0, gen_floor=0.0)
        seesaw = [
            (1, 0.95, 0.0),   # over-abstainer: no concrete generation
            (2, 0.5, 0.3),    # weak both
            (3, 0.6, 0.5),    # balanced -> should win
        ]
        for step, f1, gen in seesaw:
            params[0].data.copy_(torch.tensor([float(step)], dtype=torch.float64))
            tracker.consider(step, abstention_f1=f1, concrete_gen=gen, params=params)
        self.assertEqual(tracker.best_step, 3)
        self.assertTrue(tracker.restore(params))
        self.assertAlmostEqual(float(params[0]), 3.0)

    def test_over_abstainer_scores_below_balanced(self) -> None:
        torch = _torch_or_skip(self)
        params = [torch.tensor([0.0], dtype=torch.float64)]
        tracker = CombinedBestCheckpointTracker(f1_floor=0.0, gen_floor=0.0)
        over_score = tracker.consider(
            1, abstention_f1=0.95, concrete_gen=0.0, params=params
        )
        balanced_score = tracker.consider(
            2, abstention_f1=0.6, concrete_gen=0.5, params=params
        )
        self.assertEqual(over_score, 0.0)  # gen=0 collapses the harmonic mean
        self.assertGreater(balanced_score, 0.0)
        self.assertLess(over_score, balanced_score)  # gameability blocked
        # The over-abstainer never becomes best; the balanced step wins.
        self.assertEqual(tracker.best_step, 2)

    def test_all_below_floor_restore_returns_false(self) -> None:
        torch = _torch_or_skip(self)
        params = [torch.tensor([7.0], dtype=torch.float64)]
        tracker = CombinedBestCheckpointTracker(f1_floor=0.85, gen_floor=0.05)
        # gen below floor -> gated to 0 -> never snapshots (0.0 is not > initial None? )
        tracker.consider(1, abstention_f1=0.95, concrete_gen=0.0, params=params)
        tracker.consider(2, abstention_f1=0.5, concrete_gen=0.5, params=params)  # f1 below floor
        self.assertFalse(tracker.restore(params))
        self.assertAlmostEqual(float(params[0]), 7.0)
        self.assertIsNone(tracker.best_step)

    def test_tie_keeps_earliest_step(self) -> None:
        torch = _torch_or_skip(self)
        params = [torch.tensor([0.0], dtype=torch.float64)]
        tracker = CombinedBestCheckpointTracker(f1_floor=0.0, gen_floor=0.0)
        params[0].data.copy_(torch.tensor([1.0], dtype=torch.float64))
        tracker.consider(1, abstention_f1=0.6, concrete_gen=0.5, params=params)
        params[0].data.copy_(torch.tensor([2.0], dtype=torch.float64))
        tracker.consider(2, abstention_f1=0.6, concrete_gen=0.5, params=params)  # identical score
        self.assertEqual(tracker.best_step, 1)  # strict > keeps earliest
        tracker.restore(params)
        self.assertAlmostEqual(float(params[0]), 1.0)


if __name__ == "__main__":
    unittest.main()
