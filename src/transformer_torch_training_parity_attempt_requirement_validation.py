"""Standalone validation for PyTorch attempt next-requirements artifacts."""

from __future__ import annotations

from typing import Any

from transformer_torch_training_parity_attempt_requirements import (
    TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENTS_KIND,
    TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENTS_SCHEMA_VERSION,
)
from transformer_torch_training_parity_attempt_requirement_routing import (
    validate_torch_training_requirement_routing,
)


def validate_torch_training_parity_attempt_requirements(
    requirements: dict[str, Any],
) -> None:
    """Validate the standalone next-requirements artifact shape."""

    if not isinstance(requirements, dict):
        raise ValueError("next_requirements must be a dict")
    if (
        requirements.get("schema_version")
        != TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENTS_SCHEMA_VERSION
    ):
        raise ValueError("next_requirements.schema_version is inconsistent")
    if requirements.get("kind") != TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENTS_KIND:
        raise ValueError("next_requirements.kind is inconsistent")
    stage = _required_string(requirements, "stage")
    status = _required_string(requirements, "status")
    _require_string_list(requirements, "primary_blockers")
    _require_string_list(requirements, "next_actions")
    _require_keys(
        requirements,
        (
            "runtime_status",
            "parity_attempt_allowed",
            "training_readiness_status",
            "training_replay_parity_status",
            "training_report_passed",
        ),
    )
    validate_torch_training_requirement_routing(
        stage=stage,
        status=status,
        primary_blockers=requirements["primary_blockers"],
        next_actions=requirements["next_actions"],
        runtime_status=requirements["runtime_status"],
        exact_actions=True,
        require_blockers=True,
    )
    if stage == "complete" and (
        requirements["primary_blockers"] or requirements["next_actions"]
    ):
        raise ValueError("next_requirements.complete actions must be empty")
    if not isinstance(requirements["parity_attempt_allowed"], bool):
        raise ValueError("next_requirements.parity_attempt_allowed must be a bool")
    if not isinstance(requirements["training_report_passed"], bool):
        raise ValueError("next_requirements.training_report_passed must be a bool")


def _required_string(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"next_requirements.{key} must be a non-empty string")
    return value


def _require_string_list(record: dict[str, Any], key: str) -> None:
    value = record.get(key)
    if not isinstance(value, list):
        raise ValueError(f"next_requirements.{key} must be a list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"next_requirements.{key} must contain strings")


def _require_keys(record: dict[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        if key not in record:
            raise ValueError(f"next_requirements.{key} is missing")
