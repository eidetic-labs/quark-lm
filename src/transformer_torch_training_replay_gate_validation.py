"""Validation for aggregate PyTorch training replay parity gates."""

from __future__ import annotations

from typing import Any

from transformer_torch_training_replay_parity_gate import (
    TORCH_TRAINING_REPLAY_BLOCKED_STATUS,
    TORCH_TRAINING_REPLAY_GATE_SCHEMA_VERSION,
    TORCH_TRAINING_REPLAY_MATCHED_STATUS,
    TORCH_TRAINING_REPLAY_PENDING_STATUS,
)


TORCH_TRAINING_REPLAY_GATE_CHECKS = (
    "runtime_available",
    "runtime_kind",
    "dtype_available",
    "training_readiness",
    "initial_loss",
    "backward",
    "optimizer_step_readiness",
    "optimizer_step_control",
    "replay_control",
    "replay_gradient_signatures",
    "replay_buffer",
    "replay_update",
    "replay_final_evaluation",
    "replay_checkpoint",
)


def validate_torch_training_replay_parity_gate(gate: dict[str, Any]) -> None:
    """Validate the aggregate replay gate before trusting candidate evidence."""

    if not isinstance(gate, dict):
        raise ValueError("candidate.training_replay_parity_gate must be a dict")
    if gate.get("schema_version") != TORCH_TRAINING_REPLAY_GATE_SCHEMA_VERSION:
        raise ValueError("candidate.training_replay_parity_gate.schema_version")
    checks = _validated_checks(gate)
    if not isinstance(gate.get("passed"), bool):
        raise ValueError("candidate.training_replay_parity_gate.passed is invalid")
    if gate.get("promoted_training_backend") is not False:
        raise ValueError("candidate.training_replay_parity_gate must not promote")
    _require_non_empty_string(gate, "implementation_status")
    _require_non_empty_string(gate, "reason")
    _validate_summary(gate, checks)
    _validate_status(gate, checks)


def _validated_checks(gate: dict[str, Any]) -> list[dict[str, Any]]:
    checks = gate.get("checks")
    if not isinstance(checks, list):
        raise ValueError("candidate.training_replay_parity_gate.checks is invalid")
    if any(not isinstance(check, dict) for check in checks):
        raise ValueError("candidate.training_replay_parity_gate.checks is invalid")
    check_names = tuple(check.get("name") for check in checks)
    if check_names != TORCH_TRAINING_REPLAY_GATE_CHECKS:
        raise ValueError("candidate.training_replay_parity_gate.checks catalog")
    for check in checks:
        name = check["name"]
        if not isinstance(check.get("passed"), bool):
            raise ValueError(
                f"candidate.training_replay_parity_gate.checks.{name}.passed"
            )
    return checks


def _validate_summary(
    gate: dict[str, Any],
    checks: list[dict[str, Any]],
) -> None:
    summary = gate.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("candidate.training_replay_parity_gate.summary is invalid")
    failed = [check["name"] for check in checks if check["passed"] is not True]
    if not _matches_count(summary.get("check_count"), len(checks)):
        raise ValueError("candidate.training_replay_parity_gate.summary.check_count")
    if not _matches_count(
        summary.get("passed_check_count"),
        len(checks) - len(failed),
    ):
        raise ValueError(
            "candidate.training_replay_parity_gate.summary.passed_check_count"
        )
    if summary.get("failed_checks") != failed:
        raise ValueError("candidate.training_replay_parity_gate.summary.failed_checks")


def _validate_status(
    gate: dict[str, Any],
    checks: list[dict[str, Any]],
) -> None:
    if gate.get("passed") != all(check["passed"] for check in checks):
        raise ValueError("candidate.training_replay_parity_gate.passed is inconsistent")
    expected_status = _expected_status(checks)
    expected_parity = _expected_parity_status(checks)
    expected_implementation = _expected_implementation_status(checks)
    if gate.get("status") != expected_status:
        raise ValueError("candidate.training_replay_parity_gate.status")
    if gate.get("parity_status") != expected_parity:
        raise ValueError("candidate.training_replay_parity_gate.parity_status")
    if gate.get("implementation_status") != expected_implementation:
        raise ValueError("candidate.training_replay_parity_gate.implementation_status")


def _expected_status(checks: list[dict[str, Any]]) -> str:
    if all(check["passed"] for check in checks):
        return TORCH_TRAINING_REPLAY_MATCHED_STATUS
    if _check_passed(checks, "runtime_available") is not True:
        return TORCH_TRAINING_REPLAY_BLOCKED_STATUS
    return TORCH_TRAINING_REPLAY_PENDING_STATUS


def _expected_parity_status(checks: list[dict[str, Any]]) -> str:
    if all(check["passed"] for check in checks):
        return "matched"
    if _check_passed(checks, "runtime_available") is not True:
        return "failed"
    return "pending"


def _expected_implementation_status(checks: list[dict[str, Any]]) -> str:
    if all(check["passed"] for check in checks):
        return TORCH_TRAINING_REPLAY_MATCHED_STATUS
    if _check_passed(checks, "runtime_available") is not True:
        return "runtime_unavailable"
    return TORCH_TRAINING_REPLAY_PENDING_STATUS


def _check_passed(checks: list[dict[str, Any]], name: str) -> bool:
    return next(check["passed"] for check in checks if check["name"] == name)


def _matches_count(value: Any, expected: int) -> bool:
    return type(value) is int and value == expected


def _require_non_empty_string(record: dict[str, Any], key: str) -> None:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"candidate.training_replay_parity_gate.{key} must be a non-empty string"
        )
