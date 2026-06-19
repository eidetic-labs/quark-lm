from __future__ import annotations

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

    def test_gate_rejects_inconsistent_replay_control_counts(self) -> None:
        probe_cases = [
            ("executed_microstep_count", 1),
            ("backward_pass_count", 1),
            ("gradient_signature_match_count", 1),
            ("gradient_signature_mismatch_count", 1),
        ]
        for field, value in probe_cases:
            with self.subTest(field=field):
                probes = _matching_probes()
                probes["accumulation_replay_control_probe"][field] = value

                gate = build_torch_training_replay_parity_gate(
                    runtime=_runtime(),
                    readiness=_readiness("ready"),
                    probes=probes,
                )

                self.assertEqual(gate["status"], TORCH_TRAINING_REPLAY_PENDING_STATUS)
                self.assertFalse(gate["passed"])
                self.assertEqual(
                    gate["summary"]["failed_checks"],
                    ["replay_gradient_signatures"],
                )

    def test_gate_rejects_replay_control_without_microstep_records(self) -> None:
        probes = _matching_probes()
        probes["accumulation_replay_control_probe"]["microsteps"] = []

        gate = build_torch_training_replay_parity_gate(
            runtime=_runtime(),
            readiness=_readiness("ready"),
            probes=probes,
        )

        self.assertEqual(gate["status"], TORCH_TRAINING_REPLAY_PENDING_STATUS)
        self.assertFalse(gate["passed"])
        self.assertEqual(
            gate["summary"]["failed_checks"],
            ["replay_gradient_signatures"],
        )

    def test_gate_rejects_passed_probe_with_mismatched_status(self) -> None:
        probe_cases = [
            (
                "accumulation_replay_buffer_comparison",
                "replay_buffer_signature_mismatch",
                "replay_buffer",
            ),
            (
                "accumulation_replay_update_comparison",
                "replay_update_signature_mismatch",
                "replay_update",
            ),
            (
                "accumulation_replay_final_evaluation",
                "replay_final_evaluation_mismatch",
                "replay_final_evaluation",
            ),
            (
                "accumulation_replay_checkpoint_compatibility",
                "replay_checkpoint_mismatch",
                "replay_checkpoint",
            ),
        ]
        for probe_key, status, failed_check in probe_cases:
            with self.subTest(probe_key=probe_key):
                probes = _matching_probes()
                probes[probe_key] = {"passed": True, "status": status}

                gate = build_torch_training_replay_parity_gate(
                    runtime=_runtime(),
                    readiness=_readiness("ready"),
                    probes=probes,
                )

                self.assertEqual(gate["status"], TORCH_TRAINING_REPLAY_PENDING_STATUS)
                self.assertFalse(gate["passed"])
                self.assertEqual(gate["summary"]["failed_checks"], [failed_check])

    def test_gate_rejects_matched_probe_without_proof_flag(self) -> None:
        probe_cases = {
            "accumulation_replay_buffer_comparison": (
                "buffered_gradient_parity_proven",
                "replay_buffer",
            ),
            "accumulation_replay_update_comparison": (
                "optimizer_update_parity_proven",
                "replay_update",
            ),
            "accumulation_replay_final_evaluation": (
                "final_loss_parity_proven",
                "replay_final_evaluation",
            ),
            "accumulation_replay_checkpoint_compatibility": (
                "checkpoint_parity_proven",
                "replay_checkpoint",
            ),
        }
        for probe_key, (proof_flag, failed_check) in probe_cases.items():
            with self.subTest(probe_key=probe_key, proof_flag=proof_flag):
                probes = _matching_probes()
                probes[probe_key][proof_flag] = False

                gate = build_torch_training_replay_parity_gate(
                    runtime=_runtime(),
                    readiness=_readiness("ready"),
                    probes=probes,
                )

                self.assertEqual(gate["status"], TORCH_TRAINING_REPLAY_PENDING_STATUS)
                self.assertFalse(gate["passed"])
                self.assertEqual(gate["summary"]["failed_checks"], [failed_check])

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
            "schema_version": 1,
            "status": "accumulation_replay_control_recorded",
            "gradient_signature_match_count": 2,
            "gradient_signature_mismatch_count": 0,
            "planned_microstep_count": 2,
            "executed_microstep_count": 2,
            "backward_pass_count": 2,
            "microsteps": [{"step": 1}, {"step": 2}],
        },
        "accumulation_replay_buffer_comparison": {
            "schema_version": 1,
            "passed": True,
            "status": "replay_buffer_signature_matched",
            "buffered_gradient_parity_proven": True,
        },
        "accumulation_replay_update_comparison": {
            "schema_version": 1,
            "passed": True,
            "status": "replay_update_signature_matched",
            "optimizer_update_parity_proven": True,
        },
        "accumulation_replay_final_evaluation": {
            "schema_version": 1,
            "passed": True,
            "status": "replay_final_evaluation_matched",
            "final_logit_parity_proven": True,
            "final_loss_parity_proven": True,
        },
        "accumulation_replay_checkpoint_compatibility": {
            "schema_version": 1,
            "passed": True,
            "status": "replay_checkpoint_compatible",
            "checkpoint_parity_proven": True,
        },
    }
