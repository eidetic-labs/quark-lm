"""Experiment decision evidence for self-improvement cycles."""

from __future__ import annotations

from typing import Any


def self_improvement_decision_evidence(report: dict[str, Any]) -> list[dict[str, Any]]:
    gate = report["promotion_gate"]
    return [
        {
            "name": "training_recipe",
            "passed": "training_recipe" in report,
        },
        {
            "name": "closed_world_verifier",
            "passed": report.get("closed_world_verifier", {}).get("passed", False),
        },
        {
            "name": "constraint_first_promotion",
            "passed": report.get("constraint_first_promotion", {}).get("passed", False),
        },
        {
            "name": "admission_probe_audit",
            "passed": report["admission_probe_audit"]["passed"],
        },
        {
            "name": "glossary_probe_audit",
            "passed": report["glossary_probe_audit"]["passed"],
        },
        {
            "name": "tokenizer_candidate_guard",
            "passed": report.get("tokenizer_candidate_guard", {}).get("passed") is True,
        },
        {
            "name": "heldout_prompt_leakage",
            "passed": report["prompt_leakage_audit"]["heldout"]["passed"],
        },
        {
            "name": "owner_heldout_prompt_leakage",
            "passed": report["prompt_leakage_audit"]["owner_heldout"]["passed"],
        },
        {"name": "forgetting_audit", "passed": report["forgetting_audit"]["passed"]},
        {"name": "exact_eval_audit", "passed": report["exact_eval_audit"]["passed"]},
        {"name": "promotion_gate", "passed": gate["passed"]},
    ]


def self_improvement_experiment_decision(
    report: dict[str, Any],
) -> tuple[str, str, list[dict[str, Any]]]:
    gate = report["promotion_gate"]
    evidence = self_improvement_decision_evidence(report)
    if gate["passed"]:
        return (
            "promoted",
            "Self-improvement run passed all declared gates and is eligible for promotion.",
            evidence,
        )
    return (
        "rejected",
        "Self-improvement run was recorded as evidence but failed at least one declared gate.",
        evidence,
    )
