"""Forgetting regression audits for self-improvement reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def component_final_evals(report: dict[str, Any], component: str) -> dict[str, Any]:
    if component == "responder":
        return report.get("responder", {})
    return report.get(component, {}).get("final", {})


def _comparison_check(
    component: str,
    eval_name: str,
    previous: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, Any]:
    current_count = current.get("count", 0)
    previous_count = previous.get("count", 0)
    current_exact = current.get("exact", 0)
    previous_exact = previous.get("exact", 0)
    current_rate = current.get("exact_rate", 0.0)
    previous_rate = previous.get("exact_rate", 0.0)
    passed = (
        current_count >= previous_count
        and current_exact >= previous_exact
        and current_rate >= previous_rate
    )
    return {
        "component": component,
        "eval": eval_name,
        "status": "compared",
        "previous": {
            "count": previous_count,
            "exact": previous_exact,
            "exact_rate": previous_rate,
        },
        "current": {
            "count": current_count,
            "exact": current_exact,
            "exact_rate": current_rate,
        },
        "passed": passed,
    }


def _component_forgetting_checks(
    component: str,
    previous_report: dict[str, Any],
    current_report: dict[str, Any],
) -> list[dict[str, Any]]:
    previous_evals = component_final_evals(previous_report, component)
    current_evals = component_final_evals(current_report, component)
    checks: list[dict[str, Any]] = []
    for eval_name in sorted(set(previous_evals) | set(current_evals)):
        if eval_name not in previous_evals:
            checks.append(
                {
                    "component": component,
                    "eval": eval_name,
                    "status": "new_eval",
                    "passed": True,
                }
            )
            continue
        if eval_name not in current_evals:
            checks.append(
                {
                    "component": component,
                    "eval": eval_name,
                    "status": "missing_current_eval",
                    "passed": False,
                }
            )
            continue
        checks.append(
            _comparison_check(
                component,
                eval_name,
                previous_evals[eval_name],
                current_evals[eval_name],
            )
        )
    return checks


def audit_forgetting(
    current_report: dict[str, Any],
    previous_report_path: Path | None,
) -> dict[str, Any]:
    if previous_report_path is None:
        return {
            "mode": "previous_report",
            "compare_report": None,
            "passed": True,
            "status": "not_evaluated_no_previous_report",
            "checks": [],
        }

    previous_report = read_report(previous_report_path)
    checks: list[dict[str, Any]] = []
    for component in ("responder", "answer_model", "answer_decoder"):
        checks.extend(
            _component_forgetting_checks(component, previous_report, current_report)
        )

    return {
        "mode": "previous_report",
        "compare_report": str(previous_report_path),
        "passed": all(check["passed"] for check in checks),
        "status": "evaluated",
        "rule": "For every shared eval set, the current run must keep at least the previous count, exact matches, and exact rate.",
        "checks": checks,
    }
