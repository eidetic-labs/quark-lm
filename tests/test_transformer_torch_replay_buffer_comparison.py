from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_torch_backend import (
    TORCH_REPLAY_BUFFER_MATCHED_STATUS,
    TORCH_REPLAY_BUFFER_MISMATCH_STATUS,
    TORCH_REPLAY_BUFFER_NOT_RUN_STATUS,
    build_torch_replay_buffer_comparison,
)


class TransformerTorchReplayBufferComparisonTests(unittest.TestCase):
    def test_comparison_matches_replayed_buffer_signatures(self) -> None:
        comparison = build_torch_replay_buffer_comparison(
            fixture=_fixture(),
            replay_control_probe=_control_probe([[1.0, -1.0], [3.0, 1.0]]),
        )

        self.assertEqual(comparison["status"], TORCH_REPLAY_BUFFER_MATCHED_STATUS)
        self.assertTrue(comparison["passed"])
        self.assertEqual(comparison["matched_step_count"], 2)
        self.assertEqual(comparison["mismatched_step_count"], 0)
        self.assertEqual(comparison["update_step_count"], 1)
        self.assertTrue(comparison["step_alignment"]["passed"])
        self.assertTrue(comparison["buffered_gradient_parity_proven"])
        self.assertFalse(comparison["optimizer_update_parity_proven"])
        self.assertFalse(comparison["final_loss_parity_proven"])
        self.assertEqual(
            comparison["records"][1]["comparisons"]["accumulated_gradient"][
                "status"
            ],
            "matched",
        )
        json.dumps(comparison)

    def test_comparison_reports_replayed_buffer_mismatch(self) -> None:
        comparison = build_torch_replay_buffer_comparison(
            fixture=_fixture(),
            replay_control_probe=_control_probe([[1.0, -1.0], [0.0, 0.0]]),
        )

        self.assertEqual(comparison["status"], TORCH_REPLAY_BUFFER_MISMATCH_STATUS)
        self.assertFalse(comparison["passed"])
        self.assertGreater(comparison["mismatched_step_count"], 0)
        self.assertFalse(comparison["buffered_gradient_parity_proven"])
        self.assertEqual(
            comparison["records"][1]["comparisons"]["buffer_after_add"][
                "status"
            ],
            "mismatch",
        )

    def test_comparison_waits_for_completed_replay_control(self) -> None:
        comparison = build_torch_replay_buffer_comparison(
            fixture=_fixture(),
            replay_control_probe={"status": "not_run"},
        )

        self.assertEqual(comparison["status"], TORCH_REPLAY_BUFFER_NOT_RUN_STATUS)
        self.assertFalse(comparison["passed"])
        self.assertFalse(comparison["buffered_gradient_parity_proven"])

    def test_comparison_rejects_incomplete_replay_steps(self) -> None:
        comparison = build_torch_replay_buffer_comparison(
            fixture=_fixture(),
            replay_control_probe=_control_probe([[1.0, -1.0]]),
        )

        self.assertEqual(comparison["status"], TORCH_REPLAY_BUFFER_NOT_RUN_STATUS)
        self.assertFalse(comparison["passed"])
        self.assertFalse(comparison["step_alignment"]["passed"])
        self.assertEqual(comparison["step_alignment"]["replay_steps"], [1])
        self.assertEqual(comparison["step_alignment"]["scalar_steps"], [1, 2])

    def test_comparison_rejects_misordered_replay_steps(self) -> None:
        comparison = build_torch_replay_buffer_comparison(
            fixture=_fixture(),
            replay_control_probe=_control_probe(
                [[3.0, 1.0], [1.0, -1.0]],
                steps=[2, 1],
            ),
        )

        self.assertEqual(comparison["status"], TORCH_REPLAY_BUFFER_NOT_RUN_STATUS)
        self.assertFalse(comparison["passed"])
        self.assertFalse(comparison["step_alignment"]["passed"])


def _fixture() -> dict:
    return {
        "parameter_manifest": {"parameter_count": 2},
        "tolerance": {"absolute": 1e-9, "relative": 1e-9},
        "training_case": {
            "step_records": [
                _step_record(
                    step=1,
                    clipped=[1.0, -1.0],
                    before=[0.0, 0.0],
                    after=[1.0, -1.0],
                    accumulated=None,
                    pending_before=0,
                    update_applied=False,
                ),
                _step_record(
                    step=2,
                    clipped=[3.0, 1.0],
                    before=[1.0, -1.0],
                    after=[4.0, 0.0],
                    accumulated=[2.0, 0.0],
                    pending_before=1,
                    update_applied=True,
                ),
            ]
        },
    }


def _control_probe(
    gradients: list[list[float]],
    *,
    steps: list[int] | None = None,
) -> dict:
    steps = steps or list(range(1, len(gradients) + 1))
    return {
        "status": "accumulation_replay_control_recorded",
        "case_id": "training-01",
        "microsteps": [
            {
                "step": step,
                "gradient_snapshot": _snapshot(values),
            }
            for step, values in zip(steps, gradients)
        ],
    }


def _snapshot(values: list[float]) -> dict:
    return {
        "parameters": [
            {
                "name": "weight",
                "values": list(values),
            }
        ]
    }


def _step_record(
    *,
    step: int,
    clipped: list[float],
    before: list[float],
    after: list[float],
    accumulated: list[float] | None,
    pending_before: int,
    update_applied: bool,
) -> dict:
    return {
        "step": step,
        "optimizer_gradient_evidence": {
            "clipped_gradient": _vector(clipped),
            "buffer_before": _vector(before),
            "buffer_after_add": _vector(after),
            "accumulated_gradient": _optional_vector(accumulated),
            "pending_accumulation_before": pending_before,
            "update_applied": update_applied,
        },
    }


def _optional_vector(values: list[float] | None) -> dict:
    if values is None:
        return {
            "available": False,
            "values": [],
            "signature": _signature([]),
        }
    return {
        "available": True,
        "values": list(values),
        "signature": _signature(values),
    }


def _vector(values: list[float]) -> dict:
    return {"values": list(values), "signature": _signature(values)}


def _signature(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "sum": sum(values),
        "abs_sum": sum(abs(value) for value in values),
        "square_sum": sum(value * value for value in values),
    }


if __name__ == "__main__":
    unittest.main()
