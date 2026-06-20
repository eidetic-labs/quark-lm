from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import transformer_torch_backend  # noqa: E402
from transformer_torch_backend_training_exports import (  # noqa: E402
    __all__ as TRAINING_EXPORTS,
)


EXPECTED_TRAINING_EXPORTS = [
    "TORCH_TRAINING_ATTEMPT_HASH_ALGORITHM",
    "TORCH_TRAINING_BACKEND_NOT_PROMOTED_STATUS",
    "TORCH_TRAINING_BACKEND_PROMOTION_GATE_CHECKS",
    "TORCH_TRAINING_BACKEND_PROMOTION_GATE_SCHEMA_VERSION",
    "TORCH_TRAINING_BACKEND_PROMOTION_REQUIRED_FUTURE_GATES",
    "TORCH_TRAINING_BACKWARD_PROBE_SCHEMA_VERSION",
    "TORCH_TRAINING_BLOCKED_STATUS",
    "TORCH_TRAINING_CANDIDATE_ROUTE_FIELDS",
    "TORCH_TRAINING_CANDIDATE_RUNTIME_FIELDS",
    "TORCH_TRAINING_LOSS_PROBE_SCHEMA_VERSION",
    "TORCH_TRAINING_PARITY_ATTEMPT_AUDIT_EVIDENCE_HASH_KEYS",
    "TORCH_TRAINING_PARITY_ATTEMPT_AUDIT_KIND",
    "TORCH_TRAINING_PARITY_ATTEMPT_AUDIT_STATUSES",
    "TORCH_TRAINING_PARITY_ATTEMPT_BASE_AUDIT_KEYS",
    "TORCH_TRAINING_PARITY_ATTEMPT_INVALID_AUDIT_FORBIDDEN_FIELDS",
    "TORCH_TRAINING_PARITY_ATTEMPT_INVALID_AUDIT_KEYS",
    "TORCH_TRAINING_PARITY_ATTEMPT_INVALID_AUDIT_RESULT_KEYS",
    "TORCH_TRAINING_PARITY_ATTEMPT_VALID_AUDIT_KEYS",
    "TORCH_TRAINING_PARITY_ATTEMPT_VALID_AUDIT_RESULT_KEYS",
    "TORCH_TRAINING_PARITY_ATTEMPT_FILES",
    "TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENT_STAGES",
    "TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENTS_KIND",
    "TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENTS_SCHEMA_VERSION",
    "TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_ACTION_BY_STATUS",
    "TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_ACTIONS",
    "TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_BLOCKER_BY_STATUS",
    "TORCH_TRAINING_PARITY_ATTEMPT_MATCHED_STATUS",
    "TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_BLOCKED_FALLBACK_STATUS",
    "TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_READY_STATUS",
    "TORCH_TRAINING_PARITY_CANDIDATE_KIND",
    "TORCH_TRAINING_PARITY_CANDIDATE_SCHEMA_VERSION",
    "TORCH_TRAINING_PENDING_STATUS",
    "TORCH_TRAINING_READINESS_BASE_CHECKS",
    "TORCH_TRAINING_READINESS_CHECK_CATALOGS",
    "TORCH_TRAINING_READINESS_RUNTIME_CHECKS",
    "TORCH_TRAINING_READINESS_SCHEMA_VERSION",
    "TORCH_TRAINING_READY_STATUS",
    "TORCH_TRAINING_REPLAY_BLOCKED_STATUS",
    "TORCH_TRAINING_REPLAY_GATE_BOOLEAN_CHECKS",
    "TORCH_TRAINING_REPLAY_GATE_CHECKS",
    "TORCH_TRAINING_REPLAY_GATE_CONTROL_COUNT_CHECK",
    "TORCH_TRAINING_REPLAY_GATE_PROBE_CHECKS",
    "TORCH_TRAINING_REPLAY_GATE_SCHEMA_VERSION",
    "TORCH_TRAINING_REPLAY_GATE_STATUS_CHECKS",
    "TORCH_TRAINING_REPLAY_MATCHED_STATUS",
    "TORCH_TRAINING_REPLAY_PARITY_STATUS",
    "TORCH_TRAINING_REPLAY_PENDING_STATUS",
    "TORCH_TRAINING_RUNTIME_INCOMPLETE_STATUS",
    "TORCH_TRAINING_STATE_SCHEMA_VERSION",
    "REQUIRED_TORCH_TRAINING_CANDIDATE_KEYS",
    "REQUIRED_TORCH_TRAINING_ATTEMPT_ARTIFACTS",
    "build_torch_training_backend_promotion_gate",
    "build_torch_training_backward_probe",
    "build_torch_training_initial_loss_probe",
    "build_torch_runtime_report_hash",
    "build_torch_training_attempt_payload_hash",
    "build_torch_training_parity_attempt_compact_requirements",
    "build_torch_training_parity_attempt_audit",
    "build_torch_training_parity_attempt_hashes",
    "build_torch_training_parity_attempt_requirements",
    "build_torch_training_parity_candidate",
    "build_torch_training_readiness",
    "build_torch_training_replay_parity_gate",
    "build_torch_training_state",
    "load_torch_training_parity_attempt_artifact_set",
    "resolve_torch_training_parity_attempt_passed",
    "resolve_torch_training_parity_attempt_status",
    "summarize_torch_training_state",
    "torch_training_weights_from_state",
    "validate_torch_training_parity_attempt_requirements",
    "validate_torch_training_parity_attempt_audit",
    "validate_torch_training_parity_attempt_audit_keys",
    "validate_torch_training_parity_attempt_audit_requirements",
    "validate_torch_training_parity_attempt_audit_status",
    "validate_torch_training_parity_attempt_artifact_set",
    "validate_torch_training_parity_attempt_summaries",
    "validate_torch_training_backend_promotion_gate",
    "validate_torch_training_candidate_runtime_report",
    "validate_torch_training_candidate_routing",
    "validate_torch_training_parity_candidate",
    "validate_torch_training_readiness",
    "validate_torch_training_replay_gate_check",
    "validate_torch_training_replay_parity_gate",
    "validate_torch_training_state_summary",
]


class TransformerTorchBackendTrainingExportTests(unittest.TestCase):
    def test_training_exports_match_phase_critical_catalog(self) -> None:
        self.assertEqual(TRAINING_EXPORTS, EXPECTED_TRAINING_EXPORTS)
        self.assertEqual(len(TRAINING_EXPORTS), len(set(TRAINING_EXPORTS)))
        for export_name in EXPECTED_TRAINING_EXPORTS:
            self.assertTrue(hasattr(transformer_torch_backend, export_name))


if __name__ == "__main__":
    unittest.main()
