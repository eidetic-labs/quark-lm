from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_torch_backend import (
    TORCH_ACCUMULATION_REPLAY_CONTROL_SCHEMA_VERSION,
    TORCH_REPLAY_BUFFER_COMPARISON_SCHEMA_VERSION,
    TORCH_REPLAY_CHECKPOINT_SCHEMA_VERSION,
    TORCH_REPLAY_FINAL_EVAL_SCHEMA_VERSION,
    TORCH_REPLAY_UPDATE_COMPARISON_SCHEMA_VERSION,
    TORCH_RUNTIME_KIND_PYTORCH,
    TORCH_TRAINING_REPLAY_PENDING_STATUS,
    build_torch_training_replay_parity_gate,
)


class TransformerTorchTrainingReplayGateCountTests(unittest.TestCase):
    def test_gate_rejects_string_replay_control_counts(self) -> None:
        probes = _matching_probes()
        probes["accumulation_replay_control_probe"][
            "planned_microstep_count"
        ] = "2"

        gate = build_torch_training_replay_parity_gate(
            runtime=_runtime(),
            readiness={"status": "ready"},
            probes=probes,
        )

        check = _check(gate, "replay_gradient_signatures")
        self.assertEqual(gate["status"], TORCH_TRAINING_REPLAY_PENDING_STATUS)
        self.assertFalse(gate["passed"])
        self.assertFalse(check["passed"])
        self.assertFalse(check["count_types_valid"])

    def test_gate_rejects_boolean_replay_control_counts(self) -> None:
        probes = _matching_probes()
        probes["accumulation_replay_control_probe"][
            "gradient_signature_match_count"
        ] = True

        gate = build_torch_training_replay_parity_gate(
            runtime=_runtime(),
            readiness={"status": "ready"},
            probes=probes,
        )

        check = _check(gate, "replay_gradient_signatures")
        self.assertFalse(gate["passed"])
        self.assertFalse(check["count_types_valid"])

    def test_gate_rejects_negative_replay_control_counts(self) -> None:
        probes = _matching_probes()
        probes["accumulation_replay_control_probe"][
            "gradient_signature_mismatch_count"
        ] = -1

        gate = build_torch_training_replay_parity_gate(
            runtime=_runtime(),
            readiness={"status": "ready"},
            probes=probes,
        )

        check = _check(gate, "replay_gradient_signatures")
        self.assertFalse(gate["passed"])
        self.assertFalse(check["count_types_valid"])

    def test_gate_rejects_non_list_microstep_records(self) -> None:
        probes = _matching_probes()
        probes["accumulation_replay_control_probe"]["microsteps"] = {"step": 1}

        gate = build_torch_training_replay_parity_gate(
            runtime=_runtime(),
            readiness={"status": "ready"},
            probes=probes,
        )

        check = _check(gate, "replay_gradient_signatures")
        self.assertFalse(gate["passed"])
        self.assertFalse(check["count_types_valid"])


def _runtime() -> dict:
    return {
        "available": True,
        "runtime_kind": TORCH_RUNTIME_KIND_PYTORCH,
        "dtype_available": True,
    }


def _matching_probes() -> dict:
    return {
        "initial_loss_probe": {"status": "matched"},
        "backward_probe": {"status": "gradients_available"},
        "optimizer_step_probe": {"status": "ready_for_optimizer_execution"},
        "optimizer_step_execution_probe": {"status": "step_control_matched"},
        "accumulation_replay_control_probe": {
            "schema_version": TORCH_ACCUMULATION_REPLAY_CONTROL_SCHEMA_VERSION,
            "status": "accumulation_replay_control_recorded",
            "gradient_signature_match_count": 2,
            "gradient_signature_mismatch_count": 0,
            "planned_microstep_count": 2,
            "executed_microstep_count": 2,
            "backward_pass_count": 2,
            "microsteps": [{"step": 1}, {"step": 2}],
        },
        "accumulation_replay_buffer_comparison": {
            "schema_version": TORCH_REPLAY_BUFFER_COMPARISON_SCHEMA_VERSION,
            "passed": True,
            "status": "replay_buffer_signature_matched",
            "buffered_gradient_parity_proven": True,
        },
        "accumulation_replay_update_comparison": {
            "schema_version": TORCH_REPLAY_UPDATE_COMPARISON_SCHEMA_VERSION,
            "passed": True,
            "status": "replay_update_signature_matched",
            "optimizer_update_parity_proven": True,
        },
        "accumulation_replay_final_evaluation": {
            "schema_version": TORCH_REPLAY_FINAL_EVAL_SCHEMA_VERSION,
            "passed": True,
            "status": "replay_final_evaluation_matched",
            "final_logit_parity_proven": True,
            "final_loss_parity_proven": True,
        },
        "accumulation_replay_checkpoint_compatibility": {
            "schema_version": TORCH_REPLAY_CHECKPOINT_SCHEMA_VERSION,
            "passed": True,
            "status": "replay_checkpoint_compatible",
            "checkpoint_parity_proven": True,
        },
    }


def _check(gate: dict, name: str) -> dict:
    return next(check for check in gate["checks"] if check["name"] == name)


if __name__ == "__main__":
    unittest.main()
