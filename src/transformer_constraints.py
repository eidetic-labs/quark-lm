"""Constraint-first reports for transformer answer-training metrics."""

from __future__ import annotations

from typing import Any

from constraint_first_report import (
    build_constraint_first_promotion_report,
    promotion_check,
)


def transformer_constraint_report(metrics: dict[str, Any]) -> dict[str, Any]:
    direct_answer = metrics.get("direct_answer")
    direct_answer_present = isinstance(direct_answer, dict)
    direct_final = direct_answer.get("final", {}) if direct_answer_present else {}
    direct_baseline = direct_answer.get("baseline", {}) if direct_answer_present else {}
    branch_gate = (
        direct_answer.get("direct_answer_branch_context_gate", {})
        if direct_answer_present
        else {}
    )
    diversity = (
        direct_final.get("branch_diversity_target", {})
        if isinstance(direct_final, dict)
        else {}
    )
    coverage = _coverage_preservation_details(direct_baseline, direct_final)
    constraints = [
        promotion_check(
            "baseline_snapshot_recorded",
            bool(metrics.get("baseline")),
            "Transformer screens must record a baseline snapshot.",
        ),
        promotion_check(
            "final_snapshot_recorded",
            bool(metrics.get("final")),
            "Transformer screens must record a final snapshot.",
        ),
        promotion_check(
            "closed_world_training_data",
            metrics.get("training_data")
            == "answer_model corpus-derived AnswerExample lessons",
            "Training data must be corpus-derived AnswerExample lessons.",
            {"training_data": metrics.get("training_data")},
        ),
        promotion_check(
            "closed_world_verifier",
            metrics.get("closed_world_verifier", {}).get("passed") is True,
            "Training-plan verifier evidence must pass before promotion.",
        ),
        promotion_check(
            "no_pretrained_weights",
            metrics.get("pretrained_weights") is False,
            "Transformer promotion forbids pretrained weights.",
        ),
        promotion_check(
            "no_pretrained_tokenizer",
            metrics.get("pretrained_tokenizer") is False,
            "Transformer promotion forbids pretrained tokenizers.",
        ),
        promotion_check(
            "no_external_embeddings",
            metrics.get("external_embeddings") is False,
            "Transformer promotion forbids external embeddings.",
        ),
        promotion_check(
            "direct_answer_evidence_present",
            direct_answer_present,
            "Reliable-answer promotion requires direct-answer evidence.",
        ),
        promotion_check(
            "branch_context_gate",
            branch_gate.get("passed") is True,
            "Direct-answer branch contexts must pass semantic coverage gates.",
            {"branch_context_gate": branch_gate},
        ),
        promotion_check(
            "branch_diversity_target",
            diversity.get("passed") is True,
            "Direct-answer snapshots must pass branch diversity targets.",
            {"branch_diversity_target": diversity},
        ),
        promotion_check(
            "target_coverage_preserved",
            coverage["passed"],
            "Final branch target coverage may not regress below baseline coverage.",
            coverage,
        ),
    ]
    quality_checks = [
        promotion_check(
            "direct_greedy_exact",
            _direct_answer_final_exact(direct_final),
            "Direct greedy transformer answers must be exact across recorded eval sets.",
            _direct_answer_exact_details(direct_final),
        )
    ]
    return build_constraint_first_promotion_report(
        component="transformer-answer-train",
        run_id=str(
            metrics.get("run_id")
            or metrics.get("experiment_intent_path")
            or "transformer"
        ),
        subject_kind="transformer_answer_metrics",
        constraints=constraints,
        quality_checks=quality_checks,
        subject_path=metrics.get("metrics_path"),
    )


def _coverage_preservation_details(
    baseline: dict[str, Any],
    final: dict[str, Any],
) -> dict[str, Any]:
    baseline_coverage = (
        baseline.get("branch_target_coverage_by_profile", {})
        if isinstance(baseline, dict)
        else {}
    )
    final_coverage = (
        final.get("branch_target_coverage_by_profile", {})
        if isinstance(final, dict)
        else {}
    )
    regressions = []
    for profile, baseline_value in sorted(baseline_coverage.items()):
        final_value = final_coverage.get(profile)
        if final_value is None or float(final_value) + 1e-12 < float(baseline_value):
            regressions.append(
                {
                    "profile": profile,
                    "baseline": baseline_value,
                    "final": final_value,
                }
            )
    return {
        "passed": not regressions,
        "baseline_coverage": baseline_coverage,
        "final_coverage": final_coverage,
        "regressions": regressions,
    }


def _direct_answer_final_exact(final: dict[str, Any]) -> bool:
    details = _direct_answer_exact_details(final)
    return details["eval_set_count"] > 0 and not details["failed_eval_sets"]


def _direct_answer_exact_details(final: dict[str, Any]) -> dict[str, Any]:
    evals = final.get("evals", {}) if isinstance(final, dict) else {}
    if not isinstance(evals, dict):
        evals = {}
    failed = []
    for name, summary in sorted(evals.items()):
        if not isinstance(summary, dict):
            failed.append(name)
            continue
        if summary.get("count", 0) <= 0 or summary.get("exact") != summary.get("count"):
            failed.append(name)
    return {
        "eval_set_count": len(evals),
        "failed_eval_sets": failed,
    }
