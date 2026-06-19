"""Validation for optional PyTorch training parity attempt summaries."""

from __future__ import annotations

from typing import Any

from corpus_artifacts import SCHEMA_VERSION
from transformer_torch_training_attempt_boundary import (
    build_torch_training_attempt_boundary,
)
from transformer_torch_training_promotion_gate import (
    TORCH_TRAINING_BACKEND_NOT_PROMOTED_STATUS,
)


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
    boundary = attempt["closed_world_boundary"]
    _validate_boundary(boundary)
    _validate_promotion_gate(attempt["training_backend_promotion_gate"], boundary)
    if require_artifacts:
        _validate_artifacts(attempt.get("artifacts"))


def _validate_boundary(boundary: dict[str, Any]) -> None:
    expected = build_torch_training_attempt_boundary()
    for key, expected_value in expected.items():
        if boundary.get(key) is not expected_value:
            raise ValueError(f"closed_world_boundary.{key} is invalid")


def _validate_promotion_gate(gate: dict[str, Any], boundary: dict[str, Any]) -> None:
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
    if not isinstance(gate.get("required_future_gates"), list):
        raise ValueError("training backend promotion gate future gates missing")
    boundary_failures = _boundary_failures(boundary)
    if gate.get("closed_world_boundary_passed") is not (not boundary_failures):
        raise ValueError("training backend promotion gate boundary status is invalid")
    if gate.get("closed_world_boundary_failures") != boundary_failures:
        raise ValueError("training backend promotion gate boundary failures are invalid")


def _boundary_failures(boundary: dict[str, Any]) -> list[str]:
    expected = build_torch_training_attempt_boundary()
    return [
        key
        for key, expected_value in expected.items()
        if boundary.get(key) is not expected_value
    ]


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
