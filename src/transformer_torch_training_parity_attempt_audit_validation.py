"""Validation for compact PyTorch training parity attempt audits."""

from __future__ import annotations

from typing import Any

from corpus_artifacts import SCHEMA_VERSION
from transformer_torch_training_parity_attempt_hashes import (
    HASHED_TORCH_TRAINING_ATTEMPT_ARTIFACTS,
    TORCH_TRAINING_ATTEMPT_HASH_ALGORITHM,
)
from transformer_torch_training_parity_attempt_reader import (
    TORCH_TRAINING_PARITY_ATTEMPT_FILES,
)
from transformer_torch_training_parity_attempt_requirement_routing import (
    validate_torch_training_requirement_routing,
)
from transformer_torch_training_parity_attempt_audit_status import (
    validate_torch_training_parity_attempt_audit_status,
)
from transformer_torch_training_parity_attempt_audit_requirements import (
    validate_torch_training_parity_attempt_audit_requirements,
)
from transformer_torch_training_promotion_gate import (
    TORCH_TRAINING_BACKEND_NOT_PROMOTED_STATUS,
)


TORCH_TRAINING_PARITY_ATTEMPT_AUDIT_KIND = (
    "transformer_torch_training_parity_attempt_audit"
)
TORCH_TRAINING_PARITY_ATTEMPT_AUDIT_STATUSES = (
    "artifact_set_valid",
    "artifact_set_invalid",
)
TORCH_TRAINING_PARITY_ATTEMPT_AUDIT_EVIDENCE_HASH_KEYS = (
    "runtime_report",
    "candidate",
    "training_replay_parity_gate",
    "training_parity_report",
)


def validate_torch_training_parity_attempt_audit(
    audit: dict[str, Any],
) -> None:
    """Validate the compact audit result emitted for written attempts."""

    if not isinstance(audit, dict):
        raise ValueError("audit must be a dict")
    if audit.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("audit.schema_version is inconsistent")
    if audit.get("kind") != TORCH_TRAINING_PARITY_ATTEMPT_AUDIT_KIND:
        raise ValueError("audit.kind is inconsistent")
    _require_non_empty_string(audit, "output_dir")
    if audit.get("artifact_files") != dict(TORCH_TRAINING_PARITY_ATTEMPT_FILES):
        raise ValueError("audit.artifact_files is inconsistent")
    status = _require_non_empty_string(audit, "status")
    if status not in TORCH_TRAINING_PARITY_ATTEMPT_AUDIT_STATUSES:
        raise ValueError("audit.status is unsupported")
    _require_bool(audit, "passed")
    if status == "artifact_set_valid":
        _validate_valid_audit(audit)
        return
    _validate_invalid_audit(audit)


def _validate_valid_audit(audit: dict[str, Any]) -> None:
    if audit.get("passed") is not True:
        raise ValueError("audit.passed is inconsistent")
    for key in (
        "fixture_id",
        "attempt_status",
        "runtime_status",
        "training_readiness_status",
        "training_replay_parity_status",
        "next_requirements_stage",
        "next_requirements_status",
        "training_backend_promotion_status",
    ):
        _require_non_empty_string(audit, key)
    _require_bool(audit, "attempt_passed")
    _require_bool(audit, "parity_attempt_allowed")
    _require_bool(audit, "training_replay_parity_passed")
    _require_bool(audit, "training_report_passed")
    for key in (
        "runtime_failed_checks",
        "training_readiness_failed_checks",
        "training_replay_parity_failed_checks",
        "training_report_failed_checks",
    ):
        _require_string_list(audit, key)
    validate_torch_training_parity_attempt_audit_status(audit)
    validate_torch_training_parity_attempt_audit_requirements(audit)
    if audit.get("promoted_training_backend") is not False:
        raise ValueError("audit.promoted_training_backend must be false")
    if audit.get("artifact_hash_algorithm") != TORCH_TRAINING_ATTEMPT_HASH_ALGORITHM:
        raise ValueError("audit.artifact_hash_algorithm is inconsistent")
    _validate_hash_map(
        audit,
        "artifact_hashes",
        HASHED_TORCH_TRAINING_ATTEMPT_ARTIFACTS,
    )
    _validate_hash_map(
        audit,
        "evidence_hashes",
        TORCH_TRAINING_PARITY_ATTEMPT_AUDIT_EVIDENCE_HASH_KEYS,
    )
    _require_string_list(audit, "next_actions")
    _validate_routing(audit)
    if (
        audit.get("training_backend_promotion_status")
        != TORCH_TRAINING_BACKEND_NOT_PROMOTED_STATUS
    ):
        raise ValueError("audit.training_backend_promotion_status is inconsistent")
    if "error" in audit or "error_type" in audit:
        raise ValueError("audit.valid_result must not include errors")


def _validate_invalid_audit(audit: dict[str, Any]) -> None:
    if audit.get("passed") is not False:
        raise ValueError("audit.passed is inconsistent")
    _require_non_empty_string(audit, "error_type")
    _require_non_empty_string(audit, "error")


def _validate_hash_map(
    audit: dict[str, Any],
    key: str,
    expected_keys: tuple[str, ...],
) -> None:
    value = audit.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"audit.{key} must be a dict")
    if set(value) != set(expected_keys):
        raise ValueError(f"audit.{key} keys are inconsistent")
    for name, digest in value.items():
        if not _is_sha256(digest):
            raise ValueError(f"audit.{key}.{name} is invalid")


def _validate_routing(audit: dict[str, Any]) -> None:
    validate_torch_training_requirement_routing(
        stage=audit["next_requirements_stage"],
        status=audit["next_requirements_status"],
        next_actions=audit["next_actions"],
        runtime_status=audit["runtime_status"],
        exact_actions=False,
        require_blockers=False,
        stage_error="audit.next_requirements_stage",
        status_error="audit.next_requirements_status",
        action_error="audit.next_actions",
    )


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(char in "0123456789abcdef" for char in value)


def _require_bool(record: dict[str, Any], key: str) -> None:
    if not isinstance(record.get(key), bool):
        raise ValueError(f"audit.{key} must be a bool")


def _require_non_empty_string(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"audit.{key} must be a non-empty string")
    return value


def _require_string_list(record: dict[str, Any], key: str) -> None:
    value = record.get(key)
    if not isinstance(value, list):
        raise ValueError(f"audit.{key} must be a list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"audit.{key} must contain strings")
