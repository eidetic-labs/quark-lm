"""Exact key schema for compact PyTorch training parity attempt audits."""

from __future__ import annotations

from typing import Any


TORCH_TRAINING_PARITY_ATTEMPT_BASE_AUDIT_KEYS = (
    "schema_version",
    "kind",
    "output_dir",
    "artifact_files",
    "status",
    "passed",
)

TORCH_TRAINING_PARITY_ATTEMPT_VALID_AUDIT_RESULT_KEYS = (
    "fixture_id",
    "attempt_status",
    "attempt_passed",
    "runtime_status",
    "parity_attempt_allowed",
    "runtime_failed_checks",
    "training_readiness_status",
    "training_readiness_failed_checks",
    "training_replay_parity_status",
    "training_replay_parity_passed",
    "training_replay_parity_failed_checks",
    "training_report_passed",
    "training_report_failed_checks",
    "next_requirements_stage",
    "next_requirements_status",
    "next_actions",
    "training_backend_promotion_status",
    "promoted_training_backend",
    "artifact_hash_algorithm",
    "artifact_hashes",
    "evidence_hashes",
)

TORCH_TRAINING_PARITY_ATTEMPT_INVALID_AUDIT_RESULT_KEYS = (
    "error_type",
    "error",
)

TORCH_TRAINING_PARITY_ATTEMPT_VALID_AUDIT_KEYS = (
    *TORCH_TRAINING_PARITY_ATTEMPT_BASE_AUDIT_KEYS,
    *TORCH_TRAINING_PARITY_ATTEMPT_VALID_AUDIT_RESULT_KEYS,
)

TORCH_TRAINING_PARITY_ATTEMPT_INVALID_AUDIT_KEYS = (
    *TORCH_TRAINING_PARITY_ATTEMPT_BASE_AUDIT_KEYS,
    *TORCH_TRAINING_PARITY_ATTEMPT_INVALID_AUDIT_RESULT_KEYS,
)

TORCH_TRAINING_PARITY_ATTEMPT_INVALID_AUDIT_FORBIDDEN_FIELDS = (
    TORCH_TRAINING_PARITY_ATTEMPT_VALID_AUDIT_RESULT_KEYS
)


def validate_torch_training_parity_attempt_audit_keys(
    audit: dict[str, Any],
) -> None:
    """Reject compact audits whose payload shape drifts from the known schema."""

    expected_keys = _expected_keys_for_status(audit.get("status"))
    if expected_keys is None:
        return
    if set(audit) != set(expected_keys):
        raise ValueError("audit.keys are inconsistent")


def _expected_keys_for_status(status: Any) -> tuple[str, ...] | None:
    if status == "artifact_set_valid":
        return TORCH_TRAINING_PARITY_ATTEMPT_VALID_AUDIT_KEYS
    if status == "artifact_set_invalid":
        return TORCH_TRAINING_PARITY_ATTEMPT_INVALID_AUDIT_KEYS
    return None
