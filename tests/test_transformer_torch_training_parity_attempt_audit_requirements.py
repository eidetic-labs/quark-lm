from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_torch_training_parity_attempt_audit_requirements import (
    validate_torch_training_parity_attempt_audit_requirements,
)


class TransformerTorchTrainingParityAttemptAuditRequirementsTests(
    unittest.TestCase
):
    def test_validator_accepts_runtime_preflight_audit(self) -> None:
        validate_torch_training_parity_attempt_audit_requirements(
            _runtime_preflight()
        )

    def test_validator_rejects_stale_runtime_action(self) -> None:
        audit = _runtime_preflight()
        audit["next_actions"] = ["fix_pytorch_runtime_preflight"]

        with self.assertRaisesRegex(ValueError, "next_actions"):
            validate_torch_training_parity_attempt_audit_requirements(audit)

    def test_validator_rejects_readiness_stage_bypass(self) -> None:
        audit = _runtime_passed()
        audit["training_readiness_status"] = "pending"
        audit["training_readiness_failed_checks"] = ["adamw_optimizer"]
        audit["next_requirements_stage"] = "training_replay_parity"
        audit["next_actions"] = ["resolve_replay_gate:replay_buffer"]

        with self.assertRaisesRegex(ValueError, "next_requirements_stage"):
            validate_torch_training_parity_attempt_audit_requirements(audit)

    def test_validator_accepts_replay_pending_audit(self) -> None:
        audit = _runtime_passed()
        audit["training_replay_parity_failed_checks"] = ["replay_buffer"]

        validate_torch_training_parity_attempt_audit_requirements(audit)

    def test_validator_rejects_stale_replay_actions(self) -> None:
        audit = _runtime_passed()
        audit["training_replay_parity_failed_checks"] = ["replay_update"]

        with self.assertRaisesRegex(ValueError, "next_actions"):
            validate_torch_training_parity_attempt_audit_requirements(audit)

    def test_validator_accepts_complete_audit(self) -> None:
        audit = _runtime_passed()
        audit["training_replay_parity_status"] = "training_replay_parity_matched"
        audit["training_replay_parity_passed"] = True
        audit["training_replay_parity_failed_checks"] = []
        audit["training_report_passed"] = True
        audit["next_requirements_stage"] = "complete"
        audit["next_requirements_status"] = "satisfied"
        audit["next_actions"] = []

        validate_torch_training_parity_attempt_audit_requirements(audit)


def _runtime_preflight() -> dict:
    return {
        "runtime_status": "blocked_runtime_unavailable",
        "parity_attempt_allowed": False,
        "runtime_failed_checks": ["runtime_available"],
        "training_readiness_status": "blocked",
        "training_readiness_failed_checks": ["runtime_available"],
        "training_replay_parity_status": "training_replay_parity_pending",
        "training_replay_parity_passed": False,
        "training_replay_parity_failed_checks": ["runtime_available"],
        "training_report_passed": False,
        "training_report_failed_checks": ["runtime_report"],
        "next_requirements_stage": "runtime_preflight",
        "next_requirements_status": "blocked",
        "next_actions": ["install_real_pytorch_runtime"],
    }


def _runtime_passed() -> dict:
    return {
        "runtime_status": "passed",
        "parity_attempt_allowed": True,
        "runtime_failed_checks": [],
        "training_readiness_status": "ready",
        "training_readiness_failed_checks": [],
        "training_replay_parity_status": "training_replay_parity_pending",
        "training_replay_parity_passed": False,
        "training_replay_parity_failed_checks": ["replay_buffer"],
        "training_report_passed": False,
        "training_report_failed_checks": [],
        "next_requirements_stage": "training_replay_parity",
        "next_requirements_status": "pending",
        "next_actions": ["resolve_replay_gate:replay_buffer"],
    }


if __name__ == "__main__":
    unittest.main()
