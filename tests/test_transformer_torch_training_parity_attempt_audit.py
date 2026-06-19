from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_torch_training_parity_attempt import (
    build_torch_training_parity_attempt,
    write_torch_training_parity_attempt,
)
from transformer_torch_training_parity_attempt_audit import (
    TORCH_TRAINING_PARITY_ATTEMPT_AUDIT_KIND,
    build_torch_training_parity_attempt_audit,
)
from transformer_torch_training_parity_attempt_audit_validation import (
    validate_torch_training_parity_attempt_audit,
)


class TransformerTorchTrainingParityAttemptAuditTests(unittest.TestCase):
    def test_audit_reports_valid_written_attempt(self) -> None:
        artifacts = _artifacts()
        with tempfile.TemporaryDirectory() as temp:
            written = write_torch_training_parity_attempt(Path(temp), artifacts)

            audit = build_torch_training_parity_attempt_audit(Path(temp))

        self.assertEqual(audit["kind"], TORCH_TRAINING_PARITY_ATTEMPT_AUDIT_KIND)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["status"], "artifact_set_valid")
        self.assertEqual(audit["fixture_id"], written["fixture_id"])
        self.assertEqual(audit["attempt_status"], written["status"])
        self.assertEqual(audit["attempt_passed"], written["passed"])
        self.assertEqual(
            audit["next_requirements_stage"],
            written["next_requirements"]["stage"],
        )
        self.assertEqual(
            audit["next_actions"],
            written["next_requirements"]["next_actions"],
        )
        self.assertFalse(audit["promoted_training_backend"])
        validate_torch_training_parity_attempt_audit(audit)

    def test_audit_reports_invalid_written_attempt(self) -> None:
        artifacts = _artifacts()
        with tempfile.TemporaryDirectory() as temp:
            written = write_torch_training_parity_attempt(Path(temp), artifacts)
            attempt_path = Path(written["artifacts"]["attempt"])
            attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
            attempt["artifacts"]["candidate"] = "missing-candidate.json"
            attempt_path.write_text(
                json.dumps(attempt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            audit = build_torch_training_parity_attempt_audit(Path(temp))

        self.assertFalse(audit["passed"])
        self.assertEqual(audit["status"], "artifact_set_invalid")
        self.assertEqual(audit["error_type"], "ValueError")
        self.assertIn("artifacts.candidate", audit["error"])
        validate_torch_training_parity_attempt_audit(audit)

    def test_audit_reports_missing_attempt_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            missing = Path(temp) / "missing-attempt"

            audit = build_torch_training_parity_attempt_audit(missing)

        self.assertFalse(audit["passed"])
        self.assertEqual(audit["status"], "artifact_set_invalid")
        self.assertEqual(audit["error_type"], "ValueError")
        self.assertIn("artifacts.fixture", audit["error"])
        validate_torch_training_parity_attempt_audit(audit)

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

    def test_validator_rejects_invalid_audit_without_error_detail(self) -> None:
        audit = _invalid_audit()
        audit["error"] = ""

        with self.assertRaisesRegex(ValueError, "audit.error"):
            validate_torch_training_parity_attempt_audit(audit)

    def test_validator_rejects_drifted_artifact_file_map(self) -> None:
        audit = _valid_audit()
        audit["artifact_files"]["attempt"] = "other.json"

        with self.assertRaisesRegex(ValueError, "audit.artifact_files"):
            validate_torch_training_parity_attempt_audit(audit)


def _artifacts() -> dict:
    return build_torch_training_parity_attempt(
        corpus_dir=ROOT / "corpus",
        fixture_id="audit-training-parity-attempt",
        seed=53,
        context_index=4,
        context_size=4,
        embedding_dim=4,
        feedforward_dim=8,
        steps=2,
        importer=_missing_importer,
    )


def _missing_importer(name: str) -> object:
    raise ModuleNotFoundError(name)


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
        "training_backend_promotion_status": "not_promoted",
        "promoted_training_backend": False,
        "artifact_hash_algorithm": "sha256-json-v1",
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
