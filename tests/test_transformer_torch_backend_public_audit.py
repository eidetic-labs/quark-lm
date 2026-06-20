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
    TORCH_TRAINING_ATTEMPT_HASH_ALGORITHM,
    TORCH_TRAINING_CANDIDATE_ROUTE_FIELDS,
    TORCH_TRAINING_CANDIDATE_RUNTIME_FIELDS,
    TORCH_TRAINING_PARITY_ATTEMPT_AUDIT_KIND,
    TORCH_TRAINING_PARITY_ATTEMPT_AUDIT_EVIDENCE_HASH_KEYS,
    TORCH_TRAINING_PARITY_ATTEMPT_AUDIT_STATUSES,
    TORCH_TRAINING_PARITY_ATTEMPT_INVALID_AUDIT_FORBIDDEN_FIELDS,
    TORCH_TRAINING_PARITY_ATTEMPT_FILES,
    TORCH_TRAINING_READINESS_BASE_CHECKS,
    TORCH_TRAINING_READINESS_CHECK_CATALOGS,
    TORCH_TRAINING_PARITY_ATTEMPT_MATCHED_STATUS,
    TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_BLOCKED_FALLBACK_STATUS,
    TORCH_TRAINING_READINESS_RUNTIME_CHECKS,
    REQUIRED_TORCH_TRAINING_ATTEMPT_ARTIFACTS,
    REQUIRED_TORCH_TRAINING_CANDIDATE_KEYS,
    build_torch_runtime_report_hash,
    build_torch_training_attempt_payload_hash,
    build_torch_training_parity_attempt_compact_requirements,
    build_torch_training_parity_attempt_hashes,
    build_torch_training_parity_attempt_audit,
    load_torch_training_parity_attempt_artifact_set,
    resolve_torch_training_parity_attempt_passed,
    resolve_torch_training_parity_attempt_status,
    validate_torch_runtime_report,
    validate_torch_training_backend_promotion_gate,
    validate_torch_training_candidate_runtime_report,
    validate_torch_training_candidate_routing,
    validate_torch_training_parity_attempt_audit,
    validate_torch_training_parity_attempt_audit_requirements,
    validate_torch_training_parity_attempt_audit_status,
    validate_torch_training_parity_attempt_artifact_set,
    validate_torch_training_parity_candidate,
    validate_torch_training_readiness,
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
        self.assertTrue(
            callable(build_torch_training_parity_attempt_compact_requirements)
        )
        self.assertTrue(callable(build_torch_training_parity_attempt_audit))
        self.assertTrue(callable(resolve_torch_training_parity_attempt_passed))
        self.assertTrue(callable(resolve_torch_training_parity_attempt_status))
        self.assertTrue(callable(validate_torch_training_backend_promotion_gate))
        self.assertTrue(callable(validate_torch_training_candidate_runtime_report))
        self.assertTrue(callable(validate_torch_training_candidate_routing))
        self.assertTrue(callable(validate_torch_training_parity_attempt_audit))
        self.assertTrue(
            callable(validate_torch_training_parity_attempt_audit_requirements)
        )
        self.assertTrue(
            callable(validate_torch_training_parity_attempt_audit_status)
        )
        self.assertTrue(callable(validate_torch_training_parity_attempt_artifact_set))
        self.assertTrue(callable(validate_torch_training_parity_candidate))
        self.assertTrue(callable(validate_torch_training_readiness))
        self.assertEqual(
            REQUIRED_TORCH_TRAINING_ATTEMPT_ARTIFACTS,
            ("attempt", "fixture", "candidate", "report"),
        )
        self.assertIn("runtime_report", REQUIRED_TORCH_TRAINING_CANDIDATE_KEYS)
        self.assertIn("backend.parity_status", TORCH_TRAINING_CANDIDATE_ROUTE_FIELDS)
        self.assertEqual(
            TORCH_TRAINING_CANDIDATE_RUNTIME_FIELDS,
            ("runtime", "runtime_report"),
        )
        self.assertIn(
            TORCH_TRAINING_READINESS_BASE_CHECKS
            + TORCH_TRAINING_READINESS_RUNTIME_CHECKS,
            TORCH_TRAINING_READINESS_CHECK_CATALOGS,
        )
        self.assertEqual(
            TORCH_TRAINING_PARITY_ATTEMPT_AUDIT_KIND,
            "transformer_torch_training_parity_attempt_audit",
        )
        self.assertEqual(
            TORCH_TRAINING_PARITY_ATTEMPT_MATCHED_STATUS,
            "training_parity_matched",
        )
        self.assertEqual(
            TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_BLOCKED_FALLBACK_STATUS,
            "blocked_pytorch_runtime",
        )
        self.assertIn(
            "runtime_report",
            TORCH_TRAINING_PARITY_ATTEMPT_AUDIT_EVIDENCE_HASH_KEYS,
        )
        self.assertIn(
            "artifact_set_invalid",
            TORCH_TRAINING_PARITY_ATTEMPT_AUDIT_STATUSES,
        )
        self.assertIn(
            "evidence_hashes",
            TORCH_TRAINING_PARITY_ATTEMPT_INVALID_AUDIT_FORBIDDEN_FIELDS,
        )
        self.assertIn(
            "next_actions",
            TORCH_TRAINING_PARITY_ATTEMPT_INVALID_AUDIT_FORBIDDEN_FIELDS,
        )


if __name__ == "__main__":
    unittest.main()
