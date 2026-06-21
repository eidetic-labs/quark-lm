"""Unit test for BestCheckpointTracker (eval-driven best-checkpoint selection)."""

from __future__ import annotations

import unittest
from importlib import import_module

import support  # noqa: F401  (puts src/ on sys.path)

from transformer_best_checkpoint import BestCheckpointTracker


def _torch_or_skip(test_case: unittest.TestCase):
    try:
        return import_module("torch")
    except ModuleNotFoundError:
        test_case.skipTest("optional PyTorch runtime is not installed")
        return None


class BestCheckpointTrackerTest(unittest.TestCase):
    def test_keeps_minimum_and_restores_snapshot(self) -> None:
        torch = _torch_or_skip(self)
        params = [torch.tensor([1.0], dtype=torch.float64)]
        tracker = BestCheckpointTracker()
        tracker.consider(1, 0.5, params)  # first -> snapshot [1.0]
        params[0].data.copy_(torch.tensor([2.0], dtype=torch.float64))
        tracker.consider(2, 0.3, params)  # improves -> snapshot [2.0]
        params[0].data.copy_(torch.tensor([3.0], dtype=torch.float64))
        tracker.consider(3, 0.4, params)  # worse -> keep [2.0] / 0.3

        self.assertEqual(tracker.best_loss, 0.3)
        self.assertEqual(tracker.best_step, 2)
        # current params are [3.0]; restore writes the best snapshot [2.0] back.
        self.assertTrue(tracker.restore(params))
        self.assertAlmostEqual(float(params[0]), 2.0)

    def test_restore_without_snapshot_is_noop(self) -> None:
        torch = _torch_or_skip(self)
        params = [torch.tensor([5.0], dtype=torch.float64)]
        tracker = BestCheckpointTracker()
        self.assertFalse(tracker.restore(params))
        self.assertAlmostEqual(float(params[0]), 5.0)


if __name__ == "__main__":
    unittest.main()
