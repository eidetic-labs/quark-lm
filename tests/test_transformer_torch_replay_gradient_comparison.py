from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_torch_backend import (
    TORCH_REPLAY_GRADIENT_MATCHED_STATUS,
    TORCH_REPLAY_GRADIENT_MISMATCH_STATUS,
    build_torch_replay_gradient_comparison,
    snapshot_torch_gradients,
)


class TransformerTorchReplayGradientComparisonTests(unittest.TestCase):
    def test_snapshot_records_gradient_values_and_signature(self) -> None:
        snapshot = snapshot_torch_gradients(_state([1.0, -2.0, 0.5]))

        self.assertEqual(snapshot["gradient_tensor_count"], 1)
        self.assertEqual(snapshot["signature"]["count"], 3)
        self.assertEqual(snapshot["signature"]["sum"], -0.5)
        self.assertEqual(snapshot["parameters"][0]["values"], [1.0, -2.0, 0.5])
        json.dumps(snapshot)

    def test_comparison_matches_scalar_clipped_gradient_signature(self) -> None:
        snapshot = snapshot_torch_gradients(_state([1.0, -2.0]))

        comparison = build_torch_replay_gradient_comparison(
            scalar_step_record=_scalar_step_record([1.0, -2.0]),
            torch_gradient_snapshot=snapshot,
            tolerance={"absolute": 1e-9, "relative": 1e-9},
        )

        self.assertEqual(comparison["status"], TORCH_REPLAY_GRADIENT_MATCHED_STATUS)
        self.assertTrue(comparison["passed"])
        self.assertFalse(comparison["buffer_parity_proven"])
        json.dumps(comparison)

    def test_comparison_reports_signature_mismatch(self) -> None:
        snapshot = snapshot_torch_gradients(_state([0.0, 0.0]))

        comparison = build_torch_replay_gradient_comparison(
            scalar_step_record=_scalar_step_record([1.0, -2.0]),
            torch_gradient_snapshot=snapshot,
            tolerance={"absolute": 1e-9, "relative": 1e-9},
        )

        self.assertEqual(
            comparison["status"],
            TORCH_REPLAY_GRADIENT_MISMATCH_STATUS,
        )
        self.assertFalse(comparison["passed"])
        self.assertGreater(comparison["signature_abs_diff"]["abs_sum"], 0.0)


class _Tensor:
    def __init__(self, values: list[float]) -> None:
        self.grad = _Grad(values)


class _Grad:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return list(self._values)


def _state(values: list[float]) -> dict:
    return {
        "parameters": [
            {
                "name": "weight",
                "tensor": _Tensor(values),
            }
        ]
    }


def _scalar_step_record(values: list[float]) -> dict:
    signature = {
        "count": len(values),
        "sum": sum(values),
        "abs_sum": sum(abs(value) for value in values),
        "square_sum": sum(value * value for value in values),
    }
    return {
        "step": 1,
        "optimizer_gradient_evidence": {
            "clipped_gradient": {"signature": signature},
            "buffer_after_add": {"signature": signature},
            "accumulated_gradient": {"signature": signature},
        },
    }


if __name__ == "__main__":
    unittest.main()
