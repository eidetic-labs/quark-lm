"""Standalone validation for transformer training parity reports."""

from __future__ import annotations

from typing import Any

from transformer_training_parity_schema import (
    TRAINING_PARITY_REPORT_KIND,
    TRAINING_PARITY_SCHEMA_VERSION,
)


def validate_training_parity_report(report: dict[str, Any]) -> None:
    """Validate persisted training parity report shape and summary fields."""

    if not isinstance(report, dict):
        raise ValueError("training parity report must be a dict")
    if report.get("schema_version") != TRAINING_PARITY_SCHEMA_VERSION:
        raise ValueError("training parity report schema_version is inconsistent")
    if report.get("kind") != TRAINING_PARITY_REPORT_KIND:
        raise ValueError("training parity report kind is inconsistent")
    _require_non_empty_string(report, "fixture_id")
    _validate_candidate_backend(report)
    checks = _validated_checks(report)
    _validate_passed(report, checks)
    _validate_summary(report, checks)


def _validate_candidate_backend(report: dict[str, Any]) -> None:
    backend = report.get("candidate_backend")
    if backend is not None and (not isinstance(backend, str) or not backend.strip()):
        raise ValueError("training parity report candidate_backend is invalid")


def _validated_checks(report: dict[str, Any]) -> list[dict[str, Any]]:
    checks = report.get("checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("training parity report checks must be a non-empty list")
    if any(not isinstance(check, dict) for check in checks):
        raise ValueError("training parity report checks must contain dicts")
    names = []
    for check in checks:
        name = check.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("training parity report check name is invalid")
        if not isinstance(check.get("passed"), bool):
            raise ValueError(f"training parity report checks.{name}.passed")
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("training parity report check names must be unique")
    return checks


def _validate_passed(
    report: dict[str, Any],
    checks: list[dict[str, Any]],
) -> None:
    if not isinstance(report.get("passed"), bool):
        raise ValueError("training parity report passed must be a bool")
    if report["passed"] is not all(check["passed"] for check in checks):
        raise ValueError("training parity report passed is inconsistent")


def _validate_summary(
    report: dict[str, Any],
    checks: list[dict[str, Any]],
) -> None:
    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("training parity report summary must be a dict")
    failed = [check["name"] for check in checks if check["passed"] is not True]
    if not _matches_count(summary.get("check_count"), len(checks)):
        raise ValueError("training parity report summary.check_count")
    if not _matches_count(
        summary.get("passed_check_count"),
        len(checks) - len(failed),
    ):
        raise ValueError("training parity report summary.passed_check_count")
    if summary.get("failed_checks") != failed:
        raise ValueError("training parity report summary.failed_checks")


def _matches_count(value: Any, expected: int) -> bool:
    return type(value) is int and value == expected


def _require_non_empty_string(record: dict[str, Any], key: str) -> None:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"training parity report {key} must be a non-empty string")
