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
    TORCH_TRAINING_PARITY_ATTEMPT_AUDIT_EVIDENCE_HASH_KEYS,
    validate_torch_training_parity_attempt_audit,
)
from transformer_torch_training_promotion_gate import (
    TORCH_TRAINING_BACKEND_NOT_PROMOTED_STATUS,
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
            audit["training_replay_parity_status"],
            written["training_replay_parity_gate"]["status"],
        )
        self.assertEqual(
            audit["training_replay_parity_passed"],
            written["training_replay_parity_gate"]["passed"],
        )
        self.assertEqual(
            audit["training_report_passed"],
            written["training_parity_report"]["passed"],
        )
        self.assertEqual(
            audit["next_requirements_stage"],
            written["next_requirements"]["stage"],
        )
        self.assertEqual(
            audit["next_actions"],
            written["next_requirements"]["next_actions"],
        )
        self.assertEqual(audit["artifact_hashes"], written["artifact_hashes"])
        self.assertEqual(
            audit["evidence_hashes"],
            {
                "runtime_report": written["runtime"]["runtime_report_sha256"],
                "candidate": written["candidate"]["candidate_sha256"],
                "training_replay_parity_gate": written[
                    "training_replay_parity_gate"
                ]["training_replay_parity_gate_sha256"],
                "training_parity_report": written["training_parity_report"][
                    "training_parity_report_sha256"
                ],
            },
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
        "training_replay_parity_status": "training_replay_parity_pending",
        "training_replay_parity_passed": False,
        "training_report_passed": False,
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
