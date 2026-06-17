"""Constraint-first promotion report lifecycle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from training_recipe_validation import (
    SCHEMA_VERSION,
    require_non_empty_string,
    validate_checks,
)


CONSTRAINT_REPORT_KIND = "constraint_first_promotion_report"
QUALITY_POLICY = (
    "Loss, NLL, rank, top-k, and other quality metrics are advisory until all "
    "closed-world constraints pass."
)


def promotion_check(
    name: str,
    passed: bool,
    rule: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "status": "passed" if passed else "failed",
        "rule": rule,
        "details": dict(details or {}),
    }


def build_constraint_first_promotion_report(
    component: str,
    run_id: str,
    subject_kind: str,
    constraints: list[dict[str, Any]],
    quality_checks: list[dict[str, Any]] | None = None,
    subject_path: str | Path | None = None,
) -> dict[str, Any]:
    quality = list(quality_checks or [])
    constraints_passed = all(check.get("passed") is True for check in constraints)
    quality_metrics_considered = constraints_passed and bool(quality)
    quality_passed = (
        all(check.get("passed") is True for check in quality)
        if quality_metrics_considered
        else False
    )
    if not constraints_passed:
        status = "blocked_before_quality_metrics"
    elif not quality:
        status = "blocked_no_quality_checks"
    elif quality_passed:
        status = "eligible_for_promotion"
    else:
        status = "blocked_by_quality_checks"

    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": CONSTRAINT_REPORT_KIND,
        "component": component,
        "run_id": run_id,
        "subject_kind": subject_kind,
        "subject_path": str(subject_path) if subject_path is not None else None,
        "status": status,
        "passed": status == "eligible_for_promotion",
        "constraints_passed": constraints_passed,
        "quality_metrics_considered": quality_metrics_considered,
        "quality_metric_policy": QUALITY_POLICY,
        "constraints": [dict(check) for check in constraints],
        "quality_checks": [dict(check) for check in quality],
        "failed_constraints": [
            check["name"] for check in constraints if check.get("passed") is not True
        ],
        "failed_quality_checks": (
            [check["name"] for check in quality if check.get("passed") is not True]
            if quality_metrics_considered
            else []
        ),
        "summary": {
            "constraint_count": len(constraints),
            "passed_constraint_count": sum(
                1 for check in constraints if check.get("passed") is True
            ),
            "quality_check_count": len(quality),
            "quality_checks_considered": quality_metrics_considered,
        },
    }
    validate_constraint_first_promotion_report(report)
    return report


def validate_constraint_first_promotion_report(report: dict[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported constraint report schema_version")
    if report.get("kind") != CONSTRAINT_REPORT_KIND:
        raise ValueError(f"kind must be {CONSTRAINT_REPORT_KIND}")
    for field_name in ("component", "run_id", "subject_kind", "status"):
        require_non_empty_string(report, field_name)
    for field_name in ("passed", "constraints_passed", "quality_metrics_considered"):
        if not isinstance(report.get(field_name), bool):
            raise ValueError(f"{field_name} must be a bool")
    if report.get("quality_metric_policy") != QUALITY_POLICY:
        raise ValueError("quality_metric_policy must match the constraint-first policy")
    validate_checks(report.get("constraints"), "constraints")
    validate_checks(report.get("quality_checks"), "quality_checks")
    failed_constraints = [
        check["name"] for check in report["constraints"] if check["passed"] is not True
    ]
    if report.get("failed_constraints") != failed_constraints:
        raise ValueError("failed_constraints must match failed constraints")
    if report["quality_metrics_considered"]:
        failed_quality = [
            check["name"]
            for check in report["quality_checks"]
            if check["passed"] is not True
        ]
    else:
        failed_quality = []
    if report.get("failed_quality_checks") != failed_quality:
        raise ValueError("failed_quality_checks must match considered quality checks")
    expected_passed = report["status"] == "eligible_for_promotion"
    if report["passed"] is not expected_passed:
        raise ValueError("passed must match eligible_for_promotion status")


def constraint_first_summary(report: dict[str, Any]) -> dict[str, Any]:
    validate_constraint_first_promotion_report(report)
    return {
        "status": report["status"],
        "passed": report["passed"],
        "constraints_passed": report["constraints_passed"],
        "quality_metrics_considered": report["quality_metrics_considered"],
        "failed_constraints": list(report["failed_constraints"]),
        "failed_quality_checks": list(report["failed_quality_checks"]),
    }


def write_constraint_first_report(path: Path, report: dict[str, Any]) -> None:
    validate_constraint_first_promotion_report(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
