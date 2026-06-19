"""Validation for optional PyTorch training parity attempt summaries."""

from __future__ import annotations

from typing import Any

from corpus_artifacts import SCHEMA_VERSION
from transformer_torch_training_attempt_boundary import (
    torch_training_attempt_boundary_failures,
)
from transformer_torch_training_promotion_gate import (
    TORCH_TRAINING_BACKEND_NOT_PROMOTED_STATUS,
    TORCH_TRAINING_BACKEND_PROMOTION_GATE_CHECKS,
    TORCH_TRAINING_BACKEND_PROMOTION_GATE_SCHEMA_VERSION,
    TORCH_TRAINING_BACKEND_PROMOTION_REQUIRED_FUTURE_GATES,
)
from transformer_torch_training_parity_attempt_requirement_validation import (
    validate_torch_training_parity_attempt_requirements,
)
from transformer_torch_training_parity_attempt_summary_validation import (
    validate_torch_training_parity_attempt_summaries,
)
from transformer_torch_training_readiness import TORCH_TRAINING_READY_STATUS


TORCH_TRAINING_PARITY_ATTEMPT_KIND = "transformer_torch_training_parity_attempt"


def validate_torch_training_parity_attempt(
    attempt: dict[str, Any],
    *,
    require_artifacts: bool = False,
) -> None:
    """Validate an attempt summary before writing or trusting it as evidence."""

    if attempt.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported training parity attempt schema_version")
    if attempt.get("kind") != TORCH_TRAINING_PARITY_ATTEMPT_KIND:
        raise ValueError("invalid training parity attempt kind")
    _require_non_empty_string(attempt, "fixture_id")
    _require_non_empty_string(attempt, "status")
    _require_bool(attempt, "passed")
    if attempt.get("promoted_training_backend") is not False:
        raise ValueError("training parity attempts must not promote PyTorch")
    if attempt.get("evidence_scope") != "training_parity_attempt_only":
        raise ValueError("training parity attempt evidence_scope is invalid")
    _require_dicts(
        attempt,
        (
            "corpus",
            "runtime",
            "candidate",
            "training_replay_parity_gate",
            "training_parity_report",
            "training_backend_promotion_gate",
            "next_requirements",
            "closed_world_boundary",
        ),
    )
    validate_torch_training_parity_attempt_summaries(attempt)
    boundary = attempt["closed_world_boundary"]
    _validate_boundary(boundary)
    _validate_promotion_gate(attempt["training_backend_promotion_gate"], boundary)
    _validate_attempt_status(attempt)
    _validate_next_requirements(attempt["next_requirements"], attempt)
    if require_artifacts:
        _validate_artifacts(attempt.get("artifacts"))


def _validate_boundary(boundary: dict[str, Any]) -> None:
    failures = torch_training_attempt_boundary_failures(boundary)
    if failures:
        raise ValueError(f"closed_world_boundary.{failures[0]} is invalid")


def _validate_promotion_gate(gate: dict[str, Any], boundary: dict[str, Any]) -> None:
    if (
        gate.get("schema_version")
        != TORCH_TRAINING_BACKEND_PROMOTION_GATE_SCHEMA_VERSION
    ):
        raise ValueError("training backend promotion gate schema_version is invalid")
    if gate.get("status") != TORCH_TRAINING_BACKEND_NOT_PROMOTED_STATUS:
        raise ValueError("training backend promotion gate status is invalid")
    if gate.get("passed") is not False:
        raise ValueError("training backend promotion gate must not pass")
    if gate.get("promotion_eligible") is not False:
        raise ValueError("training backend promotion gate must not be eligible")
    if gate.get("promoted_training_backend") is not False:
        raise ValueError("training backend promotion gate must not promote")
    if gate.get("evidence_scope") != "fixture_replay_parity_only":
        raise ValueError("training backend promotion gate evidence_scope is invalid")
    _validate_promotion_gate_checks(gate)
    if gate.get("required_future_gates") != list(
        TORCH_TRAINING_BACKEND_PROMOTION_REQUIRED_FUTURE_GATES
    ):
        raise ValueError("training backend promotion gate future gates are invalid")
    boundary_failures = _boundary_failures(boundary)
    if gate.get("closed_world_boundary_passed") is not (not boundary_failures):
        raise ValueError("training backend promotion gate boundary status is invalid")
    if gate.get("closed_world_boundary_failures") != boundary_failures:
        raise ValueError("training backend promotion gate boundary failures are invalid")


def _validate_promotion_gate_checks(gate: dict[str, Any]) -> None:
    checks = gate.get("checks")
    if not isinstance(checks, list):
        raise ValueError("training backend promotion gate checks are invalid")
    if [check.get("name") for check in checks] != list(
        TORCH_TRAINING_BACKEND_PROMOTION_GATE_CHECKS
    ):
        raise ValueError("training backend promotion gate check catalog is invalid")
    for check in checks:
        if not isinstance(check.get("passed"), bool):
            raise ValueError("training backend promotion gate check status is invalid")
        if not isinstance(check.get("reason"), str) or not check["reason"].strip():
            raise ValueError("training backend promotion gate check reason is invalid")
    expected_blockers = [
        check["name"]
        for check in checks
        if check.get("passed") is False
    ]
    if gate.get("blockers") != expected_blockers:
        raise ValueError("training backend promotion gate blockers are invalid")


def _boundary_failures(boundary: dict[str, Any]) -> list[str]:
    return torch_training_attempt_boundary_failures(boundary)


def _validate_attempt_status(attempt: dict[str, Any]) -> None:
    expected_passed = attempt["training_parity_report"].get("passed")
    if attempt.get("passed") is not expected_passed:
        raise ValueError("training parity attempt passed flag is inconsistent")
    expected_status = _expected_attempt_status(attempt)
    if attempt.get("status") != expected_status:
        raise ValueError("training parity attempt status is inconsistent")


def _expected_attempt_status(attempt: dict[str, Any]) -> str:
    if attempt["training_parity_report"].get("passed") is True:
        return "training_parity_matched"
    if attempt["runtime"].get("parity_attempt_allowed") is not True:
        return str(attempt["runtime"].get("status", "blocked_pytorch_runtime"))
    return str(
        attempt["training_replay_parity_gate"].get(
            "status",
            "training_parity_pending",
        )
    )


def _validate_next_requirements(
    requirements: dict[str, Any],
    attempt: dict[str, Any],
) -> None:
    validate_torch_training_parity_attempt_requirements(requirements)
    expected_stage, expected_status = _expected_next_requirement_state(attempt)
    if requirements.get("stage") != expected_stage:
        raise ValueError("next_requirements.stage is inconsistent")
    if requirements.get("status") != expected_status:
        raise ValueError("next_requirements.status is inconsistent")
    for key, expected in _expected_next_requirement_refs(attempt).items():
        if requirements.get(key) != expected:
            raise ValueError(f"next_requirements.{key} is inconsistent")


def _expected_next_requirement_state(attempt: dict[str, Any]) -> tuple[str, str]:
    if attempt["training_parity_report"].get("passed") is True:
        return "complete", "satisfied"
    if attempt["runtime"].get("parity_attempt_allowed") is not True:
        return "runtime_preflight", "blocked"
    readiness_status = attempt["candidate"].get("training_readiness_status")
    if readiness_status != TORCH_TRAINING_READY_STATUS:
        return (
            "training_readiness",
            "blocked" if readiness_status == "blocked" else "pending",
        )
    if attempt["training_replay_parity_gate"].get("passed") is not True:
        return "training_replay_parity", "pending"
    return "training_parity_report", "pending"


def _expected_next_requirement_refs(attempt: dict[str, Any]) -> dict[str, Any]:
    return {
        "runtime_status": attempt["runtime"].get("status"),
        "parity_attempt_allowed": attempt["runtime"].get("parity_attempt_allowed"),
        "training_readiness_status": attempt["candidate"].get(
            "training_readiness_status"
        ),
        "training_replay_parity_status": attempt["training_replay_parity_gate"].get(
            "status"
        ),
        "training_report_passed": attempt["training_parity_report"].get("passed"),
    }


def _validate_artifacts(artifacts: Any) -> None:
    if not isinstance(artifacts, dict):
        raise ValueError("artifacts must be a dict")
    for key in ("fixture", "candidate", "report", "attempt"):
        value = artifacts.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"artifacts.{key} must be a non-empty string")


def _require_dicts(record: dict[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        if not isinstance(record.get(key), dict):
            raise ValueError(f"{key} must be a dict")


def _require_bool(record: dict[str, Any], key: str) -> None:
    if not isinstance(record.get(key), bool):
        raise ValueError(f"{key} must be a bool")


def _require_non_empty_string(record: dict[str, Any], key: str) -> None:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
