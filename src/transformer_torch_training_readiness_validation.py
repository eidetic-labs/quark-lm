"""Validation for PyTorch training readiness evidence."""

from __future__ import annotations

from typing import Any

from transformer_torch_training_readiness import (
    TORCH_TRAINING_BLOCKED_STATUS,
    TORCH_TRAINING_PENDING_STATUS,
    TORCH_TRAINING_READINESS_SCHEMA_VERSION,
    TORCH_TRAINING_READY_STATUS,
)


TORCH_TRAINING_READINESS_BASE_CHECKS = (
    "runtime_available",
    "dtype_available",
    "parameter_manifest",
)
TORCH_TRAINING_READINESS_RUNTIME_CHECKS = (
    "torch_tensor",
    "autograd",
    "adamw_optimizer",
)
TORCH_TRAINING_READINESS_CHECK_CATALOGS = (
    TORCH_TRAINING_READINESS_BASE_CHECKS,
    TORCH_TRAINING_READINESS_BASE_CHECKS + TORCH_TRAINING_READINESS_RUNTIME_CHECKS,
)


def validate_torch_training_readiness(readiness: dict[str, Any]) -> None:
    """Validate the standalone PyTorch training-readiness artifact."""

    if not isinstance(readiness, dict):
        raise ValueError("candidate.training_readiness must be a dict")
    if (
        readiness.get("schema_version")
        != TORCH_TRAINING_READINESS_SCHEMA_VERSION
    ):
        raise ValueError("candidate.training_readiness.schema_version is inconsistent")
    checks = _validated_checks(readiness)
    status = readiness.get("status")
    if status not in {
        TORCH_TRAINING_READY_STATUS,
        TORCH_TRAINING_PENDING_STATUS,
        TORCH_TRAINING_BLOCKED_STATUS,
    }:
        raise ValueError("candidate.training_readiness.status is invalid")
    expected_status = _expected_status(checks)
    if status != expected_status:
        raise ValueError("candidate.training_readiness.status is inconsistent")
    _validate_summary(readiness, checks)


def _validated_checks(readiness: dict[str, Any]) -> list[dict[str, Any]]:
    checks = readiness.get("checks")
    if not isinstance(checks, list):
        raise ValueError("candidate.training_readiness.checks is invalid")
    names = tuple(check.get("name") for check in checks if isinstance(check, dict))
    if names not in TORCH_TRAINING_READINESS_CHECK_CATALOGS:
        raise ValueError("candidate.training_readiness.checks catalog is inconsistent")
    for check in checks:
        if not isinstance(check, dict):
            raise ValueError("candidate.training_readiness.checks is invalid")
        name = check.get("name")
        if not isinstance(check.get("passed"), bool):
            raise ValueError(f"candidate.training_readiness.checks.{name}.passed")
    return checks


def _validate_summary(
    readiness: dict[str, Any],
    checks: list[dict[str, Any]],
) -> None:
    summary = readiness.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("candidate.training_readiness.summary is invalid")
    failed = [check["name"] for check in checks if check["passed"] is not True]
    if not _matches_count(summary.get("check_count"), len(checks)):
        raise ValueError("candidate.training_readiness.summary.check_count")
    if not _matches_count(
        summary.get("passed_check_count"),
        len(checks) - len(failed),
    ):
        raise ValueError("candidate.training_readiness.summary.passed_check_count")
    if summary.get("failed_checks") != failed:
        raise ValueError("candidate.training_readiness.summary.failed_checks")


def _expected_status(checks: list[dict[str, Any]]) -> str:
    if all(check["passed"] for check in checks):
        return TORCH_TRAINING_READY_STATUS
    if checks[0]["passed"] is not True:
        return TORCH_TRAINING_BLOCKED_STATUS
    return TORCH_TRAINING_PENDING_STATUS


def _matches_count(value: Any, expected: int) -> bool:
    return type(value) is int and value == expected
