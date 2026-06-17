"""Closed-world verifier report construction and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
REPORT_KIND = "closed_world_verifier_report"
PASS = "passed"
FAIL = "failed"


def verifier_check(
    name: str,
    passed: bool,
    rule: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": PASS if passed else FAIL,
        "passed": bool(passed),
        "rule": rule,
        "details": dict(details or {}),
    }


def verifier_report(
    component: str,
    run_id: str,
    subject_kind: str,
    checks: list[dict[str, Any]],
    subject_path: Path | str | None = None,
    related_reports: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    failed = [check["name"] for check in checks if not check.get("passed")]
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "component": component,
        "run_id": run_id,
        "subject_kind": subject_kind,
        "subject_path": str(subject_path) if subject_path is not None else None,
        "passed": not failed,
        "failed_checks": failed,
        "summary": {
            "check_count": len(checks),
            "passed_count": len(checks) - len(failed),
            "failed_count": len(failed),
        },
        "uses_external_model": False,
        "verifier_type": "deterministic_closed_world",
        "checks": checks,
        "related_reports": list(related_reports or []),
    }
    validate_verifier_report(report)
    return report


def verifier_report_summary(report: dict[str, Any]) -> dict[str, Any]:
    validate_verifier_report(report)
    return {
        "status": PASS if report["passed"] else FAIL,
        "passed": report["passed"],
        "failed_checks": list(report["failed_checks"]),
        "check_count": report["summary"]["check_count"],
        "passed_count": report["summary"]["passed_count"],
        "failed_count": report["summary"]["failed_count"],
        "uses_external_model": False,
        "verifier_type": report["verifier_type"],
    }


def validate_verifier_report(report: dict[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported verifier schema_version")
    if report.get("kind") != REPORT_KIND:
        raise ValueError(f"kind must be {REPORT_KIND}")
    for field_name in ("component", "run_id", "subject_kind", "verifier_type"):
        value = report.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")
    if not isinstance(report.get("uses_external_model"), bool):
        raise ValueError("uses_external_model must be a bool")
    checks = report.get("checks")
    if not isinstance(checks, list):
        raise ValueError("checks must be a list")
    for check in checks:
        _validate_check(check)
    failed = [check["name"] for check in checks if not check["passed"]]
    if report.get("failed_checks") != failed:
        raise ValueError("failed_checks must match failed checks")
    if report.get("passed") != (not failed):
        raise ValueError("passed must match checks")
    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("summary must be a dict")
    if summary.get("check_count") != len(checks):
        raise ValueError("summary check_count must match checks")
    if summary.get("failed_count") != len(failed):
        raise ValueError("summary failed_count must match failed checks")
    if summary.get("passed_count") != len(checks) - len(failed):
        raise ValueError("summary passed_count must match passed checks")
    related = report.get("related_reports")
    if not isinstance(related, list):
        raise ValueError("related_reports must be a list")


def write_verifier_report(path: Path, report: dict[str, Any]) -> None:
    validate_verifier_report(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


def attach_verifier_summary(
    training_plan: dict[str, Any],
    report: dict[str, Any],
    verifier_path: Path,
) -> dict[str, Any]:
    updated = dict(training_plan)
    updated["closed_world_verifier"] = {
        "status": "written",
        "path": str(verifier_path),
        "summary": verifier_report_summary(report),
    }
    return updated


def _validate_check(check: dict[str, Any]) -> None:
    for field_name in ("name", "status", "rule"):
        value = check.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"check {field_name} must be a non-empty string")
    if not isinstance(check.get("passed"), bool):
        raise ValueError("check passed must be a bool")
    expected_status = PASS if check["passed"] else FAIL
    if check["status"] != expected_status:
        raise ValueError("check status must match passed")
    if not isinstance(check.get("details"), dict):
        raise ValueError("check details must be a dict")
