"""Constraint-first reports for self-improvement cycles."""

from __future__ import annotations

from typing import Any

from constraint_first_report import (
    build_constraint_first_promotion_report,
    promotion_check,
)


def self_improvement_constraint_report(report: dict[str, Any]) -> dict[str, Any]:
    prompt_leakage = report.get("prompt_leakage_audit", {})
    constraints = [
        promotion_check(
            "closed_world_verifier",
            report.get("closed_world_verifier", {}).get("passed") is True,
            "Training-plan verifier evidence must pass before promotion.",
        ),
        promotion_check(
            "admission_probe_audit",
            report.get("admission_probe_audit", {}).get("passed") is True,
            "Generated admission probes must pass.",
        ),
        promotion_check(
            "glossary_probe_audit",
            report.get("glossary_probe_audit", {}).get("passed") is True,
            "Generated glossary probes must pass.",
        ),
        promotion_check(
            "tokenizer_candidate_guard",
            report.get("tokenizer_candidate_guard", {}).get("passed") is True,
            "Tokenizer candidates must be corpus-only, round-trip safe, and candidate-only.",
        ),
        promotion_check(
            "heldout_prompt_leakage",
            prompt_leakage.get("heldout", {}).get("passed") is True,
            "Heldout prompts must not appear in training lessons.",
        ),
        promotion_check(
            "owner_heldout_prompt_leakage",
            prompt_leakage.get("owner_heldout", {}).get("passed") is True,
            "Protected owner heldout prompts must not appear in training lessons.",
        ),
        promotion_check(
            "forgetting_audit",
            report.get("forgetting_audit", {}).get("passed") is True,
            "Current evals may not regress against the comparison report.",
        ),
    ]
    quality_checks = [
        promotion_check(
            "exact_eval_audit",
            report.get("exact_eval_audit", {}).get("passed") is True,
            "Responder, answer model, and decoder evals must be exact.",
        )
    ]
    return build_constraint_first_promotion_report(
        component="self-improvement-answer-cycle",
        run_id=str(
            report.get("run_id")
            or report.get("attempt", {}).get("path")
            or report.get("cycle", "answer")
        ),
        subject_kind="self_improvement_report",
        constraints=constraints,
        quality_checks=quality_checks,
    )
