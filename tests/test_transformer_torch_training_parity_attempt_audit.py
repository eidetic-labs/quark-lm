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
from transformer_torch_training_parity_attempt_audit_validation import (  # noqa: E402
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
            audit["runtime_failed_checks"],
            written["runtime"]["failed_checks"],
        )
        self.assertEqual(
            audit["training_readiness_status"],
            written["candidate"]["training_readiness_status"],
        )
        self.assertEqual(
            audit["training_readiness_failed_checks"],
            written["candidate"]["training_readiness_failed_checks"],
        )
        self.assertEqual(
            audit["training_replay_parity_status"],
            written["training_replay_parity_gate"]["status"],
        )
        self.assertEqual(
            audit["training_replay_parity_passed"],
            written["training_replay_parity_gate"]["passed"],
        )
        self.assertEqual(
            audit["training_replay_parity_failed_checks"],
            written["training_replay_parity_gate"]["failed_checks"],
        )
        self.assertEqual(
            audit["training_report_passed"],
            written["training_parity_report"]["passed"],
        )
        self.assertEqual(
            audit["training_report_failed_checks"],
            written["training_parity_report"]["failed_checks"],
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
