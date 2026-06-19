from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_torch_backend import (
    TORCH_ACCUMULATION_PENDING_STATUS,
    TORCH_ACCUMULATION_READY_STATUS,
    build_torch_accumulation_readiness,
)
from transformer_training_parity import build_gradient_accumulation_contract


class TransformerTorchAccumulationReadinessTests(unittest.TestCase):
    def test_clipped_multistep_accumulation_requires_buffer(self) -> None:
        readiness = build_torch_accumulation_readiness(
            contract=_contract(accumulation_steps=2, step_count=2, clip=5.0),
        )

        self.assertEqual(readiness["status"], TORCH_ACCUMULATION_PENDING_STATUS)
        self.assertTrue(readiness["requires_replayed_backward_passes"])
        self.assertTrue(readiness["requires_clipped_gradient_buffer"])
        self.assertFalse(readiness["native_loss_scaling_sufficient"])
        self.assertEqual(
            readiness["missing_requirements"],
            [
                "replay_backward_per_microstep",
                "clipped_gradient_buffer",
                "mean_gradient_reduction",
            ],
        )
        json.dumps(readiness)

    def test_unclipped_multistep_accumulation_can_use_loss_scaling(self) -> None:
        readiness = build_torch_accumulation_readiness(
            contract=_contract(accumulation_steps=4, step_count=4, clip=0.0),
        )

        self.assertEqual(readiness["status"], TORCH_ACCUMULATION_PENDING_STATUS)
        self.assertFalse(readiness["requires_clipped_gradient_buffer"])
        self.assertTrue(readiness["native_loss_scaling_sufficient"])
        self.assertEqual(
            readiness["missing_requirements"],
            [
                "replay_backward_per_microstep",
                "scale_loss_by_accumulation_steps",
                "mean_gradient_reduction",
            ],
        )

    def test_single_step_accumulation_needs_no_extra_runtime(self) -> None:
        readiness = build_torch_accumulation_readiness(
            contract=_contract(accumulation_steps=1, step_count=1, clip=5.0),
        )

        self.assertEqual(readiness["status"], TORCH_ACCUMULATION_READY_STATUS)
        self.assertEqual(readiness["missing_requirements"], [])
        self.assertFalse(readiness["requires_replayed_backward_passes"])
        self.assertFalse(readiness["requires_clipped_gradient_buffer"])


def _contract(*, accumulation_steps: int, step_count: int, clip: float) -> dict:
    return {
        "gradient_accumulation_steps": accumulation_steps,
        "gradient_accumulation": build_gradient_accumulation_contract(
            optimizer_config={
                "gradient_accumulation_steps": accumulation_steps,
                "gradient_clip": clip,
            },
        ),
        "expected_step_records": _step_records(
            accumulation_steps=accumulation_steps,
            step_count=step_count,
        ),
    }


def _step_records(*, accumulation_steps: int, step_count: int) -> list[dict]:
    records = []
    pending = 0
    for step in range(1, step_count + 1):
        pending += 1
        update_applied = pending >= accumulation_steps
        if update_applied:
            pending = 0
        records.append({"step": step, "update_applied": update_applied})
    return records


if __name__ == "__main__":
    unittest.main()
