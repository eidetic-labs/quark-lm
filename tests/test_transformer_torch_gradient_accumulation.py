from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_torch_backend import (
    TORCH_GRADIENT_ACCUMULATION_RECORDED_STATUS,
    build_torch_gradient_accumulation_report,
)


class TransformerTorchGradientAccumulationTests(unittest.TestCase):
    def test_report_marks_accumulated_gradient_parity_unproven(self) -> None:
        report = build_torch_gradient_accumulation_report(
            state=_state(),
            contract=_contract(accumulation_steps=2, step_count=2),
        )

        self.assertEqual(
            report["status"],
            TORCH_GRADIENT_ACCUMULATION_RECORDED_STATUS,
        )
        self.assertEqual(report["accumulation_steps"], 2)
        self.assertEqual(report["expected_step_count"], 2)
        self.assertEqual(report["pending_step_count"], 1)
        self.assertEqual(report["expected_update_count"], 1)
        self.assertTrue(report["uses_single_gradient_sample"])
        self.assertTrue(report["requires_replayed_backward_passes"])
        self.assertFalse(report["accumulated_gradient_parity_proven"])
        self.assertEqual(report["accumulated_gradient_parity_status"], "not_proven")
        self.assertEqual(report["current_gradient_signature"]["scalar_count"], 3)
        json.dumps(report)

    def test_report_marks_single_step_accumulation_as_not_required(self) -> None:
        report = build_torch_gradient_accumulation_report(
            state=_state(),
            contract=_contract(accumulation_steps=1, step_count=1),
        )

        self.assertFalse(report["requires_replayed_backward_passes"])
        self.assertFalse(report["accumulated_gradient_parity_proven"])
        self.assertEqual(
            report["accumulated_gradient_parity_status"],
            "not_required",
        )
        self.assertEqual(report["pending_step_count"], 0)
        self.assertEqual(report["expected_update_count"], 1)


class _Tensor:
    def __init__(self, grad: object | None) -> None:
        self.grad = grad


class _Grad:
    def __init__(self, value: object) -> None:
        self._value = value

    def tolist(self) -> object:
        return self._value


def _state() -> dict:
    return {
        "parameters": [
            {"name": "weight", "tensor": _Tensor(_Grad([1.0, -2.0]))},
            {"name": "bias", "tensor": _Tensor(_Grad([0.5]))},
            {"name": "unused", "tensor": _Tensor(None)},
        ]
    }


def _contract(*, accumulation_steps: int, step_count: int) -> dict:
    records = []
    pending = 0
    update_count = 0
    for step in range(1, step_count + 1):
        before_pending = pending
        before_update_count = update_count
        pending += 1
        update_applied = pending >= accumulation_steps
        if update_applied:
            pending = 0
            update_count += 1
        records.append(
            {
                "step": step,
                "update_applied": update_applied,
                "update_count_before": before_update_count,
                "update_count_after": update_count,
                "pending_accumulation_before": before_pending,
                "pending_accumulation_after": pending,
                "effective_learning_rate": 0.01,
            }
        )
    return {
        "gradient_source": "tensor.grad",
        "gradient_accumulation_steps": accumulation_steps,
        "expected_step_records": records,
        "expected_final_optimizer_state": {
            "update_count": update_count,
            "pending_accumulation": pending,
            "param_count": 3,
        },
    }


if __name__ == "__main__":
    unittest.main()
