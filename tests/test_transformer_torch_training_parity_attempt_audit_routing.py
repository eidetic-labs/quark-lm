from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_torch_training_parity_attempt_audit import (
    TORCH_TRAINING_PARITY_ATTEMPT_AUDIT_KIND,
)
from transformer_torch_training_parity_attempt_audit_validation import (
    validate_torch_training_parity_attempt_audit,
)
from transformer_torch_training_promotion_gate import (
    TORCH_TRAINING_BACKEND_NOT_PROMOTED_STATUS,
)


class TransformerTorchTrainingParityAttemptAuditRoutingTests(unittest.TestCase):
    def test_validator_rejects_unsupported_next_requirements_stage(self) -> None:
        audit = _valid_audit()
        audit["next_requirements_stage"] = "unknown"

        with self.assertRaisesRegex(ValueError, "next_requirements_stage"):
            validate_torch_training_parity_attempt_audit(audit)

    def test_validator_rejects_wrong_status_for_stage(self) -> None:
        audit = _valid_audit()
        audit["next_requirements_stage"] = "complete"
        audit["next_requirements_status"] = "pending"
        audit["next_actions"] = []

        with self.assertRaisesRegex(ValueError, "next_requirements_status"):
            validate_torch_training_parity_attempt_audit(audit)

    def test_validator_rejects_complete_audit_with_next_actions(self) -> None:
        audit = _valid_audit()
        audit["next_requirements_stage"] = "complete"
        audit["next_requirements_status"] = "satisfied"
        audit["next_actions"] = ["resolve_replay_gate:replay_buffer"]

        with self.assertRaisesRegex(ValueError, "next_actions"):
            validate_torch_training_parity_attempt_audit(audit)

    def test_validator_rejects_runtime_preflight_action_mismatch(self) -> None:
        audit = _valid_audit()
        audit["runtime_status"] = "blocked_dtype_unavailable"

        with self.assertRaisesRegex(ValueError, "next_actions"):
            validate_torch_training_parity_attempt_audit(audit)

    def test_validator_rejects_stage_action_mismatch(self) -> None:
        audit = _valid_audit()
        audit["next_requirements_stage"] = "training_replay_parity"
        audit["next_requirements_status"] = "pending"
        audit["runtime_status"] = "passed"
        audit["parity_attempt_allowed"] = True
        audit["next_actions"] = ["satisfy_training_readiness:replay_buffer"]

        with self.assertRaisesRegex(ValueError, "next_actions"):
            validate_torch_training_parity_attempt_audit(audit)

    def test_validator_rejects_empty_stage_action_target(self) -> None:
        audit = _valid_audit()
        audit["next_requirements_stage"] = "training_replay_parity"
        audit["next_requirements_status"] = "pending"
        audit["runtime_status"] = "passed"
        audit["parity_attempt_allowed"] = True
        audit["next_actions"] = ["resolve_replay_gate:"]

        with self.assertRaisesRegex(ValueError, "next_actions"):
            validate_torch_training_parity_attempt_audit(audit)

    def test_validator_accepts_complete_routing(self) -> None:
        audit = _valid_audit()
        audit["next_requirements_stage"] = "complete"
        audit["next_requirements_status"] = "satisfied"
        audit["runtime_status"] = "passed"
        audit["parity_attempt_allowed"] = True
        audit["next_actions"] = []

        validate_torch_training_parity_attempt_audit(audit)

    def test_validator_rejects_stale_promotion_status(self) -> None:
        audit = _valid_audit()
        audit["training_backend_promotion_status"] = "promoted"

        with self.assertRaisesRegex(ValueError, "promotion_status"):
            validate_torch_training_parity_attempt_audit(audit)


def _valid_audit() -> dict:
    return {
        "schema_version": 1,
        "kind": TORCH_TRAINING_PARITY_ATTEMPT_AUDIT_KIND,
        "output_dir": "build/attempt",
        "artifact_files": {
            "fixture": "scalar_training_fixture.json",
            "candidate": "torch_training_candidate.json",
            "report": "training_parity_report.json",
            "attempt": "torch_training_parity_attempt.json",
        },
        "status": "artifact_set_valid",
        "passed": True,
        "fixture_id": "fixture",
        "attempt_status": "blocked_runtime_unavailable",
        "attempt_passed": False,
        "runtime_status": "blocked_runtime_unavailable",
        "parity_attempt_allowed": False,
        "next_requirements_stage": "runtime_preflight",
        "next_requirements_status": "blocked",
        "next_actions": ["install_real_pytorch_runtime"],
        "training_backend_promotion_status": TORCH_TRAINING_BACKEND_NOT_PROMOTED_STATUS,
        "promoted_training_backend": False,
        "artifact_hash_algorithm": "sha256-json-v1",
        "artifact_hashes": {
            "fixture": "a" * 64,
            "candidate": "b" * 64,
            "report": "c" * 64,
        },
        "evidence_hashes": {
            "runtime_report": "d" * 64,
            "candidate": "e" * 64,
            "training_replay_parity_gate": "f" * 64,
            "training_parity_report": "0" * 64,
        },
    }


if __name__ == "__main__":
    unittest.main()
