"""Promotion checks for self-improvement reports."""

from __future__ import annotations

from typing import Any


def audit_exact_promotion(report: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for component, evals in {
        "responder": report.get("responder", {}),
        "answer_model": report.get("answer_model", {}).get("final", {}),
        "answer_decoder": report.get("answer_decoder", {}).get("final", {}),
    }.items():
        for eval_name, metrics in sorted(evals.items()):
            count = metrics.get("count", 0)
            exact = metrics.get("exact", 0)
            checks.append(
                {
                    "component": component,
                    "eval": eval_name,
                    "count": count,
                    "exact": exact,
                    "passed": count > 0 and exact == count,
                }
            )
    return {
        "passed": all(check["passed"] for check in checks),
        "rule": "Every responder, classifier, and decoder eval must be non-empty and exact before promotion.",
        "checks": checks,
    }


def _required_promotion_checks(report: dict[str, Any]) -> list[dict[str, Any]]:
    prompt_leakage = report["prompt_leakage_audit"]
    return [
        {
            "name": "admission_probe_audit",
            "passed": report["admission_probe_audit"]["passed"],
        },
        {
            "name": "glossary_probe_audit",
            "passed": report["glossary_probe_audit"]["passed"],
        },
        {
            "name": "heldout_prompt_leakage",
            "passed": prompt_leakage["heldout"]["passed"],
        },
        {
            "name": "owner_heldout_prompt_leakage",
            "passed": prompt_leakage["owner_heldout"]["passed"],
        },
        {
            "name": "forgetting_audit",
            "passed": report["forgetting_audit"]["passed"],
        },
        {
            "name": "exact_eval_audit",
            "passed": report["exact_eval_audit"]["passed"],
        },
    ]


def _optional_promotion_checks(report: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    if "closed_world_verifier" in report:
        checks.append(
            {
                "name": "closed_world_verifier",
                "passed": report["closed_world_verifier"]["passed"],
            }
        )
    if "constraint_first_promotion" in report:
        checks.append(
            {
                "name": "constraint_first_promotion",
                "passed": report["constraint_first_promotion"]["passed"],
            }
        )
    return checks


def promotion_gate(report: dict[str, Any]) -> dict[str, Any]:
    checks = [
        *_required_promotion_checks(report),
        *_optional_promotion_checks(report),
    ]
    return {
        "passed": all(check["passed"] for check in checks),
        "rule": "Promote only when generated probes, verifier, constraint-first checks, prompt leakage, forgetting, and exact eval audits all pass.",
        "checks": checks,
    }
