from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import transformer_torch_backend
from transformer_torch_backend_core_exports import __all__ as CORE_EXPORTS
from transformer_torch_backend_replay_exports import __all__ as REPLAY_EXPORTS
from transformer_torch_backend_training_exports import __all__ as TRAINING_EXPORTS
from transformer_torch_backend import (
    TORCH_RUNTIME_REPORT_CHECKS,
    TORCH_RUNTIME_REPORT_EVIDENCE_SCOPE,
    TORCH_RUNTIME_REPORT_STATUSES,
    TORCH_TRAINING_BACKEND_PROMOTION_GATE_CHECKS,
    TORCH_TRAINING_BACKEND_PROMOTION_REQUIRED_FUTURE_GATES,
    TORCH_TRAINING_ATTEMPT_HASH_ALGORITHM,
    TORCH_TRAINING_PARITY_ATTEMPT_AUDIT_KIND,
    TORCH_TRAINING_PARITY_ATTEMPT_AUDIT_EVIDENCE_HASH_KEYS,
    TORCH_TRAINING_PARITY_ATTEMPT_AUDIT_STATUSES,
    TORCH_TRAINING_PARITY_ATTEMPT_FILES,
    TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENT_STAGES,
    TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_ACTION_BY_STATUS,
    TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_BLOCKER_BY_STATUS,
    TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_ACTIONS,
    TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENTS_KIND,
    TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENTS_SCHEMA_VERSION,
    REQUIRED_TORCH_TRAINING_CANDIDATE_KEYS,
    build_torch_runtime_report_hash,
    build_torch_training_attempt_payload_hash,
    build_torch_training_parity_attempt_hashes,
    build_torch_training_parity_attempt_audit,
    build_torch_training_parity_attempt_requirements,
    load_torch_training_parity_attempt_artifact_set,
    validate_torch_runtime_report,
    validate_torch_training_backend_promotion_gate,
    validate_torch_training_parity_attempt_audit,
    validate_torch_training_parity_attempt_requirements,
    validate_torch_training_parity_attempt_summaries,
    validate_torch_training_parity_candidate,
)


class TransformerTorchBackendPublicAuditTests(unittest.TestCase):
    def test_public_backend_exports_are_grouped_without_duplicates(self) -> None:
        grouped_exports = [
            *CORE_EXPORTS,
            *REPLAY_EXPORTS,
            *TRAINING_EXPORTS,
        ]

        self.assertEqual(transformer_torch_backend.__all__, grouped_exports)
        self.assertEqual(len(grouped_exports), len(set(grouped_exports)))
        for export_name in grouped_exports:
            self.assertTrue(hasattr(transformer_torch_backend, export_name))

    def test_runtime_report_validation_contract_is_public(self) -> None:
        self.assertTrue(callable(validate_torch_runtime_report))
        self.assertIn("blocked_runtime_unavailable", TORCH_RUNTIME_REPORT_STATUSES)
        self.assertEqual(
            TORCH_RUNTIME_REPORT_CHECKS,
            ("runtime_available", "runtime_kind", "dtype_available"),
        )
        self.assertEqual(TORCH_RUNTIME_REPORT_EVIDENCE_SCOPE, "runtime_preflight_only")

    def test_training_attempt_audit_helpers_are_public(self) -> None:
        artifacts = {
            "fixture": {"kind": "fixture"},
            "candidate": {"kind": "candidate"},
            "report": {"kind": "report"},
        }

        hashes = build_torch_training_parity_attempt_hashes(artifacts)

        self.assertEqual(
            TORCH_TRAINING_ATTEMPT_HASH_ALGORITHM,
            "sha256-json-v1",
        )
        self.assertEqual(
            TORCH_TRAINING_PARITY_ATTEMPT_FILES["attempt"],
            "torch_training_parity_attempt.json",
        )
        self.assertEqual(
            set(hashes),
            {"fixture", "candidate", "report"},
        )
        self.assertTrue(callable(load_torch_training_parity_attempt_artifact_set))
        self.assertTrue(callable(build_torch_runtime_report_hash))
        self.assertTrue(callable(build_torch_training_attempt_payload_hash))
        self.assertTrue(callable(build_torch_training_parity_attempt_audit))
        self.assertTrue(callable(validate_torch_training_backend_promotion_gate))
        self.assertTrue(callable(validate_torch_training_parity_attempt_audit))
        self.assertTrue(callable(validate_torch_training_parity_candidate))
        self.assertIn("runtime_report", REQUIRED_TORCH_TRAINING_CANDIDATE_KEYS)
        self.assertEqual(
            TORCH_TRAINING_PARITY_ATTEMPT_AUDIT_KIND,
            "transformer_torch_training_parity_attempt_audit",
        )
        self.assertIn(
            "runtime_report",
            TORCH_TRAINING_PARITY_ATTEMPT_AUDIT_EVIDENCE_HASH_KEYS,
        )
        self.assertIn(
            "artifact_set_invalid",
            TORCH_TRAINING_PARITY_ATTEMPT_AUDIT_STATUSES,
        )

    def test_training_attempt_requirements_contract_is_public(self) -> None:
        requirements = build_torch_training_parity_attempt_requirements(
            runtime_report={"status": "passed", "parity_attempt_allowed": True},
            candidate={
                "training_readiness": {"status": "ready"},
                "training_replay_parity_gate": {
                    "status": "training_replay_parity_matched",
                    "passed": True,
                },
            },
            report={"passed": True},
        )

        self.assertEqual(
            requirements["kind"],
            TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENTS_KIND,
        )
        self.assertEqual(
            requirements["schema_version"],
            TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENTS_SCHEMA_VERSION,
        )
        self.assertIn("complete", TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENT_STAGES)
        self.assertIn(
            "install_real_pytorch_runtime",
            TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_ACTIONS,
        )
        self.assertEqual(
            TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_ACTION_BY_STATUS[
                "blocked_runtime_unavailable"
            ],
            "install_real_pytorch_runtime",
        )
        self.assertEqual(
            TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_BLOCKER_BY_STATUS[
                "blocked_runtime_unavailable"
            ],
            "runtime_available",
        )
        self.assertIn(
            "model_quality_gate",
            TORCH_TRAINING_BACKEND_PROMOTION_GATE_CHECKS,
        )
        self.assertIn(
            "general_training_backend_gate",
            TORCH_TRAINING_BACKEND_PROMOTION_REQUIRED_FUTURE_GATES,
        )
        validate_torch_training_parity_attempt_requirements(requirements)

    def test_training_attempt_summary_validation_is_public(self) -> None:
        attempt = {
            "corpus": {
                "corpus_dir": "corpus",
                "train_sha256": "a" * 64,
                "train_chars": 1,
                "manifest": {},
            },
            "runtime": {
                "status": "passed",
                "passed": True,
                "parity_attempt_allowed": True,
                "runtime_kind": "pytorch",
                "device": "cpu",
                "dtype": "float64",
                "runtime_report_sha256": "a" * 64,
            },
            "candidate": {
                "implementation_status": "training_replay_parity_matched",
                "parity_status": "matched",
                "training_readiness_status": "ready",
                "training_case_status": "computed",
                "candidate_sha256": "b" * 64,
            },
            "training_replay_parity_gate": {
                "status": "training_replay_parity_matched",
                "passed": True,
                "failed_checks": [],
                "training_replay_parity_gate_sha256": "c" * 64,
            },
            "training_parity_report": {
                "passed": True,
                "failed_checks": [],
                "training_parity_report_sha256": "d" * 64,
            },
        }

        validate_torch_training_parity_attempt_summaries(attempt)


if __name__ == "__main__":
    unittest.main()
