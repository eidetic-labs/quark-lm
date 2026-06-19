"""Standalone validation for PyTorch attempt next-requirements artifacts."""

from __future__ import annotations

from typing import Any

from transformer_torch_training_parity_attempt_requirements import (
    TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENT_STAGES,
    TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_ACTION_BY_STATUS,
    TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_BLOCKER_BY_STATUS,
    TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_ACTIONS,
    TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENTS_KIND,
    TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENTS_SCHEMA_VERSION,
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
    if stage not in TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENT_STAGES:
        raise ValueError("next_requirements.stage is unsupported")
    if status not in _allowed_statuses(stage):
        raise ValueError("next_requirements.status is unsupported")
    _require_string_list(requirements, "primary_blockers")
    _require_string_list(requirements, "next_actions")
    if stage == "complete" and (
        requirements["primary_blockers"] or requirements["next_actions"]
    ):
        raise ValueError("next_requirements.complete actions must be empty")
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
    _validate_stage_routing(
        stage=stage,
        primary_blockers=requirements["primary_blockers"],
        next_actions=requirements["next_actions"],
        runtime_status=requirements["runtime_status"],
    )
    if not isinstance(requirements["parity_attempt_allowed"], bool):
        raise ValueError("next_requirements.parity_attempt_allowed must be a bool")
    if not isinstance(requirements["training_report_passed"], bool):
        raise ValueError("next_requirements.training_report_passed must be a bool")


def _validate_stage_routing(
    *,
    stage: str,
    primary_blockers: list[str],
    next_actions: list[str],
    runtime_status: Any,
) -> None:
    if stage == "complete":
        return
    if not primary_blockers:
        raise ValueError("next_requirements.primary_blockers must not be empty")
    if not next_actions:
        raise ValueError("next_requirements.next_actions must not be empty")
    if stage == "runtime_preflight":
        if len(next_actions) != 1:
            raise ValueError("next_requirements.runtime action count is invalid")
        if next_actions[0] not in TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_ACTIONS:
            raise ValueError("next_requirements.runtime action is unsupported")
        expected_action = TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_ACTION_BY_STATUS.get(
            runtime_status,
            "fix_pytorch_runtime_preflight",
        )
        if next_actions[0] != expected_action:
            raise ValueError("next_requirements.runtime action is inconsistent")
        expected_blocker = TORCH_TRAINING_PARITY_ATTEMPT_RUNTIME_BLOCKER_BY_STATUS.get(
            runtime_status,
        )
        if expected_blocker is not None and expected_blocker not in primary_blockers:
            raise ValueError("next_requirements.runtime blocker is inconsistent")
        return
    expected_actions = [
        f"{_action_prefix(stage)}:{blocker}" for blocker in primary_blockers
    ]
    if next_actions != expected_actions:
        raise ValueError("next_requirements.next_actions do not match stage blockers")


def _allowed_statuses(stage: str) -> tuple[str, ...]:
    if stage == "complete":
        return ("satisfied",)
    if stage == "runtime_preflight":
        return ("blocked",)
    if stage == "training_readiness":
        return ("blocked", "pending")
    return ("pending",)


def _action_prefix(stage: str) -> str:
    if stage == "training_readiness":
        return "satisfy_training_readiness"
    if stage == "training_replay_parity":
        return "resolve_replay_gate"
    if stage == "training_parity_report":
        return "resolve_training_parity_check"
    raise ValueError("next_requirements.stage is unsupported")


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
