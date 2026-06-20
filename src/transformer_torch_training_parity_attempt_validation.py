"""Validation for optional PyTorch training parity attempt summaries."""

from __future__ import annotations

from typing import Any

from corpus_artifacts import SCHEMA_VERSION
from transformer_torch_training_attempt_boundary import (
    torch_training_attempt_boundary_failures,
)
from transformer_torch_training_parity_attempt_compact_requirements import (
    build_torch_training_parity_attempt_compact_requirements,
)
from transformer_torch_training_parity_attempt_requirement_validation import (
    validate_torch_training_parity_attempt_requirements,
)
from transformer_torch_training_parity_attempt_summary_validation import (
    validate_torch_training_parity_attempt_summaries,
)
from transformer_torch_training_parity_attempt_status import (
    resolve_torch_training_parity_attempt_passed,
    resolve_torch_training_parity_attempt_status,
)
from transformer_torch_training_promotion_gate_validation import (
    validate_torch_training_backend_promotion_gate,
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
    validate_torch_training_parity_attempt_summaries(attempt)
    boundary = attempt["closed_world_boundary"]
    _validate_boundary(boundary)
    validate_torch_training_backend_promotion_gate(
        attempt["training_backend_promotion_gate"],
        closed_world_boundary=boundary,
    )
    _validate_attempt_status(attempt)
    _validate_next_requirements(attempt["next_requirements"], attempt)
    if require_artifacts:
        _validate_artifacts(attempt.get("artifacts"))


def _validate_boundary(boundary: dict[str, Any]) -> None:
    failures = torch_training_attempt_boundary_failures(boundary)
    if failures:
        raise ValueError(f"closed_world_boundary.{failures[0]} is invalid")


def _validate_attempt_status(attempt: dict[str, Any]) -> None:
    expected_passed = resolve_torch_training_parity_attempt_passed(
        runtime=attempt["runtime"],
        training_replay_parity_gate=attempt["training_replay_parity_gate"],
        training_parity_report=attempt["training_parity_report"],
    )
    if attempt.get("passed") is not expected_passed:
        raise ValueError("training parity attempt passed flag is inconsistent")
    expected_status = resolve_torch_training_parity_attempt_status(
        runtime=attempt["runtime"],
        training_replay_parity_gate=attempt["training_replay_parity_gate"],
        training_parity_report=attempt["training_parity_report"],
    )
    if attempt.get("status") != expected_status:
        raise ValueError("training parity attempt status is inconsistent")


def _validate_next_requirements(
    requirements: dict[str, Any],
    attempt: dict[str, Any],
) -> None:
    validate_torch_training_parity_attempt_requirements(requirements)
    expected_requirements = build_torch_training_parity_attempt_compact_requirements(
        runtime=attempt["runtime"],
        candidate=attempt["candidate"],
        training_replay_parity_gate=attempt["training_replay_parity_gate"],
        training_parity_report=attempt["training_parity_report"],
    )
    for key in _NEXT_REQUIREMENT_COMPARISON_ORDER:
        expected = expected_requirements[key]
        if requirements.get(key) != expected:
            raise ValueError(f"next_requirements.{key} is inconsistent")


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


_NEXT_REQUIREMENT_COMPARISON_ORDER = (
    "schema_version",
    "kind",
    "stage",
    "status",
    "runtime_status",
    "parity_attempt_allowed",
    "training_readiness_status",
    "training_replay_parity_status",
    "training_report_passed",
    "primary_blockers",
    "next_actions",
)
