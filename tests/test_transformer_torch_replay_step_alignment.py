from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_torch_replay_step_alignment import (
    TORCH_REPLAY_STEP_ALIGNMENT_MATCHED_STATUS,
    TORCH_REPLAY_STEP_ALIGNMENT_MISMATCH_STATUS,
    build_torch_replay_step_alignment,
)


class TransformerTorchReplayStepAlignmentTests(unittest.TestCase):
    def test_alignment_matches_exact_replay_steps(self) -> None:
        alignment = build_torch_replay_step_alignment(
            replay_control_probe=_probe([1, 2]),
            scalar_step_records=_records([1, 2]),
        )

        self.assertEqual(alignment["status"], TORCH_REPLAY_STEP_ALIGNMENT_MATCHED_STATUS)
        self.assertTrue(alignment["passed"])
        self.assertEqual(alignment["replay_step_count"], 2)
        self.assertEqual(alignment["scalar_step_count"], 2)

    def test_alignment_rejects_missing_replay_steps(self) -> None:
        alignment = build_torch_replay_step_alignment(
            replay_control_probe=_probe([1]),
            scalar_step_records=_records([1, 2]),
        )

        self.assertEqual(alignment["status"], TORCH_REPLAY_STEP_ALIGNMENT_MISMATCH_STATUS)
        self.assertFalse(alignment["passed"])
        self.assertEqual(alignment["replay_steps"], [1])
        self.assertEqual(alignment["scalar_steps"], [1, 2])

    def test_alignment_rejects_misordered_replay_steps(self) -> None:
        alignment = build_torch_replay_step_alignment(
            replay_control_probe=_probe([2, 1]),
            scalar_step_records=_records([1, 2]),
        )

        self.assertEqual(alignment["status"], TORCH_REPLAY_STEP_ALIGNMENT_MISMATCH_STATUS)
        self.assertFalse(alignment["passed"])


def _probe(steps: list[int]) -> dict:
    return {
        "microsteps": [
            {
                "step": step,
            }
            for step in steps
        ],
    }


def _records(steps: list[int]) -> list[dict]:
    return [{"step": step} for step in steps]


if __name__ == "__main__":
    unittest.main()
