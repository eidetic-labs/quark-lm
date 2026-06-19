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


class TransformerTorchTrainingReplayGateSchemaTests(unittest.TestCase):
    def test_gate_rejects_replay_control_schema_mismatch(self) -> None:
        probes = _matching_probes()
        probes["accumulation_replay_control_probe"]["schema_version"] = 999

        gate = build_torch_training_replay_parity_gate(
            runtime=_runtime(),
            readiness={"status": "ready"},
            probes=probes,
        )

        check = _check(gate, "replay_gradient_signatures")
        self.assertEqual(gate["status"], TORCH_TRAINING_REPLAY_PENDING_STATUS)
        self.assertFalse(gate["passed"])
        self.assertFalse(check["passed"])
        self.assertEqual(check["expected_schema_version"], 1)
        self.assertEqual(check["schema_version"], 999)

    def test_gate_rejects_replay_probe_schema_mismatch(self) -> None:
        probe_cases = {
            "accumulation_replay_buffer_comparison": "replay_buffer",
            "accumulation_replay_update_comparison": "replay_update",
            "accumulation_replay_final_evaluation": "replay_final_evaluation",
            "accumulation_replay_checkpoint_compatibility": "replay_checkpoint",
        }
        for probe_key, failed_check in probe_cases.items():
            with self.subTest(probe_key=probe_key):
                probes = _matching_probes()
                probes[probe_key]["schema_version"] = 999

                gate = build_torch_training_replay_parity_gate(
                    runtime=_runtime(),
                    readiness={"status": "ready"},
                    probes=probes,
                )

                check = _check(gate, failed_check)
                self.assertEqual(gate["status"], TORCH_TRAINING_REPLAY_PENDING_STATUS)
                self.assertFalse(gate["passed"])
                self.assertFalse(check["passed"])
                self.assertEqual(check["schema_version"], 999)


def _runtime() -> dict:
    return {
        "available": True,
        "runtime_kind": TORCH_RUNTIME_KIND_PYTORCH,
        "dtype_available": True,
        "device": "cpu",
        "dtype": "float32",
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
