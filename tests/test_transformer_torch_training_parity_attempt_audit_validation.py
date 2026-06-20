from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_torch_training_parity_attempt_audit import (  # noqa: E402
    TORCH_TRAINING_PARITY_ATTEMPT_AUDIT_KIND,
)
from transformer_torch_training_parity_attempt_audit_validation import (  # noqa: E402
    TORCH_TRAINING_PARITY_ATTEMPT_AUDIT_EVIDENCE_HASH_KEYS,
    TORCH_TRAINING_PARITY_ATTEMPT_INVALID_AUDIT_FORBIDDEN_FIELDS,
    validate_torch_training_parity_attempt_audit,
)
from transformer_torch_training_promotion_gate import (  # noqa: E402
    TORCH_TRAINING_BACKEND_NOT_PROMOTED_STATUS,
)


class TransformerTorchTrainingParityAttemptAuditValidationTests(unittest.TestCase):
    def test_validator_rejects_inconsistent_valid_status(self) -> None:
        audit = _valid_audit()
        audit["passed"] = False

        with self.assertRaisesRegex(ValueError, "audit.passed"):
            validate_torch_training_parity_attempt_audit(audit)

    def test_validator_rejects_error_fields_on_valid_audit(self) -> None:
        audit = _valid_audit()
        audit["error"] = "stale"

        with self.assertRaisesRegex(ValueError, "valid_result"):
            validate_torch_training_parity_attempt_audit(audit)

    def test_validator_rejects_stale_attempt_status(self) -> None:
        audit = _valid_audit()
        audit["attempt_status"] = "training_parity_matched"

        with self.assertRaisesRegex(ValueError, "attempt_status"):
            validate_torch_training_parity_attempt_audit(audit)

    def test_validator_rejects_stale_attempt_passed(self) -> None:
        audit = _valid_audit()
        audit["attempt_passed"] = True

        with self.assertRaisesRegex(ValueError, "attempt_passed"):
            validate_torch_training_parity_attempt_audit(audit)

    def test_validator_rejects_invalid_audit_without_error_detail(self) -> None:
        audit = _invalid_audit()
        audit["error"] = ""

        with self.assertRaisesRegex(ValueError, "audit.error"):
            validate_torch_training_parity_attempt_audit(audit)

    def test_validator_rejects_evidence_hashes_on_invalid_audit(self) -> None:
        audit = _invalid_audit()
        audit["evidence_hashes"] = {
            "runtime_report": "d" * 64,
            "candidate": "e" * 64,
            "training_replay_parity_gate": "f" * 64,
            "training_parity_report": "0" * 64,
        }

        with self.assertRaisesRegex(ValueError, "audit.evidence_hashes"):
            validate_torch_training_parity_attempt_audit(audit)

    def test_validator_rejects_routing_actions_on_invalid_audit(self) -> None:
        audit = _invalid_audit()
        audit["next_actions"] = ["install_real_pytorch_runtime"]

        with self.assertRaisesRegex(ValueError, "audit.next_actions"):
            validate_torch_training_parity_attempt_audit(audit)

    def test_validator_rejects_drifted_artifact_file_map(self) -> None:
        audit = _valid_audit()
        audit["artifact_files"]["attempt"] = "other.json"

        with self.assertRaisesRegex(ValueError, "audit.artifact_files"):
            validate_torch_training_parity_attempt_audit(audit)

    def test_validator_rejects_missing_artifact_hash(self) -> None:
        audit = _valid_audit()
        audit["artifact_hashes"].pop("candidate")

        with self.assertRaisesRegex(ValueError, "artifact_hashes"):
            validate_torch_training_parity_attempt_audit(audit)

    def test_validator_rejects_malformed_evidence_hash(self) -> None:
        audit = _valid_audit()
        audit["evidence_hashes"]["runtime_report"] = "not-a-hash"

        with self.assertRaisesRegex(ValueError, "evidence_hashes.runtime_report"):
            validate_torch_training_parity_attempt_audit(audit)

    def test_validator_rejects_runtime_allowed_with_failures(self) -> None:
        audit = _valid_audit()
        audit["runtime_status"] = "ready_for_pytorch_parity"
        audit["parity_attempt_allowed"] = True

        with self.assertRaisesRegex(ValueError, "audit.runtime_failed_checks"):
            validate_torch_training_parity_attempt_audit(audit)

    def test_validator_rejects_ready_readiness_with_failures(self) -> None:
        audit = _valid_audit()
        audit["training_readiness_status"] = "ready"

        with self.assertRaisesRegex(
            ValueError,
            "audit.training_readiness_failed_checks",
        ):
            validate_torch_training_parity_attempt_audit(audit)

    def test_validator_rejects_passed_replay_with_failures(self) -> None:
        audit = _valid_audit()
        audit["training_replay_parity_passed"] = True

        with self.assertRaisesRegex(
            ValueError,
            "audit.training_replay_parity_failed_checks",
        ):
            validate_torch_training_parity_attempt_audit(audit)

    def test_validator_rejects_passed_report_with_failures(self) -> None:
        audit = _valid_audit()
        audit["training_report_passed"] = True

        with self.assertRaisesRegex(ValueError, "audit.training_report_failed_checks"):
            validate_torch_training_parity_attempt_audit(audit)

    def test_validator_exposes_evidence_hash_key_catalog(self) -> None:
        self.assertEqual(
            TORCH_TRAINING_PARITY_ATTEMPT_AUDIT_EVIDENCE_HASH_KEYS,
            (
                "runtime_report",
                "candidate",
                "training_replay_parity_gate",
                "training_parity_report",
            ),
        )

    def test_validator_exposes_invalid_audit_forbidden_fields(self) -> None:
        self.assertIn(
            "evidence_hashes",
            TORCH_TRAINING_PARITY_ATTEMPT_INVALID_AUDIT_FORBIDDEN_FIELDS,
        )
        self.assertIn(
            "next_actions",
            TORCH_TRAINING_PARITY_ATTEMPT_INVALID_AUDIT_FORBIDDEN_FIELDS,
        )
        self.assertIn(
            "promoted_training_backend",
            TORCH_TRAINING_PARITY_ATTEMPT_INVALID_AUDIT_FORBIDDEN_FIELDS,
        )


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


def _invalid_audit() -> dict:
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
        "status": "artifact_set_invalid",
        "passed": False,
        "error_type": "ValueError",
        "error": "attempt.artifacts is inconsistent",
    }
