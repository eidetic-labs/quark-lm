from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_torch_backend import (
    TORCH_RUNTIME_KIND_PYTORCH,
    TORCH_RUNTIME_KIND_TEST_DOUBLE,
    TORCH_TRAINING_REPLAY_BLOCKED_STATUS,
    TORCH_TRAINING_REPLAY_MATCHED_STATUS,
    TORCH_TRAINING_REPLAY_PENDING_STATUS,
    build_torch_training_replay_parity_gate,
)


class TransformerTorchTrainingReplayParityGateTests(unittest.TestCase):
    def test_gate_matches_when_all_replay_evidence_matches(self) -> None:
        gate = build_torch_training_replay_parity_gate(
            runtime=_runtime(),
            readiness=_readiness("ready"),
            probes=_matching_probes(),
        )

        self.assertEqual(gate["status"], TORCH_TRAINING_REPLAY_MATCHED_STATUS)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["parity_status"], "matched")
        self.assertEqual(gate["implementation_status"], gate["status"])
        self.assertFalse(gate["promoted_training_backend"])
        self.assertEqual(gate["summary"]["failed_checks"], [])
        json.dumps(gate, sort_keys=True)

    def test_gate_stays_pending_when_a_replay_check_fails(self) -> None:
        probes = _matching_probes()
        probes["accumulation_replay_buffer_comparison"] = {
            "passed": False,
            "status": "replay_buffer_signature_mismatch",
        }

        gate = build_torch_training_replay_parity_gate(
            runtime=_runtime(),
            readiness=_readiness("ready"),
            probes=probes,
        )

        self.assertEqual(gate["status"], TORCH_TRAINING_REPLAY_PENDING_STATUS)
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["parity_status"], "pending")
        self.assertEqual(gate["summary"]["failed_checks"], ["replay_buffer"])

    def test_gate_rejects_test_double_runtime(self) -> None:
        gate = build_torch_training_replay_parity_gate(
            runtime=_runtime(runtime_kind=TORCH_RUNTIME_KIND_TEST_DOUBLE),
            readiness=_readiness("ready"),
            probes=_matching_probes(),
        )

        self.assertEqual(gate["status"], TORCH_TRAINING_REPLAY_PENDING_STATUS)
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["summary"]["failed_checks"], ["runtime_kind"])

    def test_gate_blocks_when_runtime_is_unavailable(self) -> None:
        gate = build_torch_training_replay_parity_gate(
            runtime=_runtime(available=False, dtype_available=False),
            readiness=_readiness("blocked"),
            probes={},
        )

        self.assertEqual(gate["status"], TORCH_TRAINING_REPLAY_BLOCKED_STATUS)
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["parity_status"], "failed")
        self.assertEqual(gate["implementation_status"], "runtime_unavailable")
        self.assertIn("runtime_available", gate["summary"]["failed_checks"])


def _runtime(
    *,
    available: bool = True,
    dtype_available: bool = True,
    runtime_kind: str = TORCH_RUNTIME_KIND_PYTORCH,
) -> dict:
    return {
        "available": available,
        "runtime_kind": runtime_kind,
        "dtype_available": dtype_available,
        "device": "cpu",
        "dtype": "float32",
    }


def _readiness(status: str) -> dict:
    return {"status": status}


def _matching_probes() -> dict:
    return {
        "initial_loss_probe": {"status": "matched"},
        "backward_probe": {"status": "gradients_available"},
        "optimizer_step_probe": {"status": "ready_for_optimizer_execution"},
        "optimizer_step_execution_probe": {"status": "step_control_matched"},
        "accumulation_replay_control_probe": {
            "status": "accumulation_replay_control_recorded",
            "gradient_signature_match_count": 2,
            "gradient_signature_mismatch_count": 0,
            "planned_microstep_count": 2,
        },
        "accumulation_replay_buffer_comparison": {
            "passed": True,
            "status": "replay_buffer_signature_matched",
        },
        "accumulation_replay_update_comparison": {
            "passed": True,
            "status": "replay_update_signature_matched",
        },
        "accumulation_replay_final_evaluation": {
            "passed": True,
            "status": "replay_final_evaluation_matched",
        },
        "accumulation_replay_checkpoint_compatibility": {
            "passed": True,
            "status": "replay_checkpoint_compatible",
        },
    }


if __name__ == "__main__":
    unittest.main()
