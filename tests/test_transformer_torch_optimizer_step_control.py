from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_torch_optimizer_step_control import (
    TORCH_OPTIMIZER_STEP_CONTROL_MATCHED_STATUS,
    execute_torch_optimizer_step_records,
    final_torch_optimizer_state,
    torch_optimizer_step_control_status,
    torch_step_records_match_contract,
)


class TransformerTorchOptimizerStepControlTests(unittest.TestCase):
    def test_replay_records_preserve_scalar_optimizer_control(self) -> None:
        optimizer = _FakeOptimizer()
        contract = _contract()

        records = execute_torch_optimizer_step_records(
            optimizer=optimizer,
            contract=contract,
        )
        final_state = final_torch_optimizer_state(
            contract=contract,
            step_records=records,
        )

        self.assertEqual(optimizer.param_groups[0]["lr"], 0.02)
        self.assertEqual(optimizer.step_calls, 1)
        self.assertEqual(optimizer.zero_grad_calls, 1)
        self.assertFalse(records[0]["optimizer_step_called"])
        self.assertTrue(records[1]["optimizer_step_called"])
        self.assertTrue(
            torch_step_records_match_contract(
                contract=contract,
                step_records=records,
            )
        )
        self.assertEqual(final_state, contract["expected_final_optimizer_state"])
        self.assertEqual(
            torch_optimizer_step_control_status(
                contract=contract,
                step_records=records,
                final_state=final_state,
            ),
            TORCH_OPTIMIZER_STEP_CONTROL_MATCHED_STATUS,
        )

    def test_status_reports_record_mismatch_before_final_state_mismatch(self) -> None:
        contract = _contract()
        records = execute_torch_optimizer_step_records(
            optimizer=_FakeOptimizer(),
            contract=contract,
        )
        records[1]["optimizer_step_called"] = False

        self.assertFalse(
            torch_step_records_match_contract(
                contract=contract,
                step_records=records,
            )
        )
        self.assertEqual(
            torch_optimizer_step_control_status(
                contract=contract,
                step_records=records,
                final_state={"update_count": 999},
            ),
            "step_record_mismatch",
        )

    def test_status_reports_final_state_mismatch_after_records_match(self) -> None:
        contract = _contract()
        records = execute_torch_optimizer_step_records(
            optimizer=_FakeOptimizer(),
            contract=contract,
        )

        self.assertEqual(
            torch_optimizer_step_control_status(
                contract=contract,
                step_records=records,
                final_state={"update_count": 0, "pending_accumulation": 0},
            ),
            "final_state_mismatch",
        )


class _FakeOptimizer:
    def __init__(self) -> None:
        self.param_groups = [{"lr": 0.0}]
        self.step_calls = 0
        self.zero_grad_calls = 0

    def step(self) -> None:
        self.step_calls += 1

    def zero_grad(self) -> None:
        self.zero_grad_calls += 1


def _contract() -> dict:
    return {
        "parameter_count": 3,
        "expected_final_optimizer_state": {
            "update_count": 1,
            "pending_accumulation": 0,
            "param_count": 3,
        },
        "expected_step_records": [
            {
                "step": 1,
                "effective_learning_rate": 0.01,
                "update_applied": False,
                "update_count_after": 0,
                "pending_accumulation_after": 1,
            },
            {
                "step": 2,
                "effective_learning_rate": 0.02,
                "update_applied": True,
                "update_count_after": 1,
                "pending_accumulation_after": 0,
            },
        ],
    }
