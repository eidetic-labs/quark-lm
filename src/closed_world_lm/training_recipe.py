"""Training recipe and constraint-first promotion artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
TRAINING_RECIPE_KIND = "training_recipe"
CONSTRAINT_REPORT_KIND = "constraint_first_promotion_report"
QUALITY_POLICY = (
    "Loss, NLL, rank, top-k, and other quality metrics are advisory until all "
    "closed-world constraints pass."
)


def build_training_recipe(
    version: str,
    component: str,
    run_id: str,
    recipe_id: str,
    purpose: str,
    model: dict[str, Any],
    tokenizer: dict[str, Any],
    data: dict[str, Any],
    objective: dict[str, Any],
    optimizer: dict[str, Any],
    artifacts: list[str | Path],
    gates: list[dict[str, Any]],
    replay: dict[str, Any] | None = None,
    rerun: dict[str, Any] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    recipe = {
        "schema_version": SCHEMA_VERSION,
        "kind": TRAINING_RECIPE_KIND,
        "version": version,
        "component": component,
        "run_id": run_id,
        "recipe_id": recipe_id,
        "purpose": purpose,
        "uses_external_model": False,
        "model": dict(model),
        "tokenizer": dict(tokenizer),
        "data": dict(data),
        "objective": dict(objective),
        "optimizer": dict(optimizer),
        "replay": dict(replay or {}),
        "artifacts": [str(path) for path in artifacts],
        "gates": [dict(gate) for gate in gates],
        "rerun": dict(rerun or {}),
        "notes": list(notes or []),
    }
    validate_training_recipe(recipe)
    return recipe


def validate_training_recipe(recipe: dict[str, Any]) -> None:
    if recipe.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported training recipe schema_version")
    if recipe.get("kind") != TRAINING_RECIPE_KIND:
        raise ValueError(f"kind must be {TRAINING_RECIPE_KIND}")
    for field_name in ("version", "component", "run_id", "recipe_id", "purpose"):
        _require_non_empty_string(recipe, field_name)
    if recipe.get("uses_external_model") is not False:
        raise ValueError("training recipes must not use an external model")
    for field_name in ("model", "tokenizer", "data", "objective", "optimizer", "replay", "rerun"):
        if not isinstance(recipe.get(field_name), dict):
            raise ValueError(f"{field_name} must be a dict")
    _require_string_list(recipe, "artifacts")
    _validate_named_rules(recipe.get("gates"), "gates")
    if not isinstance(recipe.get("notes"), list):
        raise ValueError("notes must be a list")


def training_recipe_summary(recipe: dict[str, Any]) -> dict[str, Any]:
    validate_training_recipe(recipe)
    return {
        "recipe_id": recipe["recipe_id"],
        "version": recipe["version"],
        "component": recipe["component"],
        "uses_external_model": False,
        "artifact_count": len(recipe["artifacts"]),
        "gate_count": len(recipe["gates"]),
        "replay_status": recipe["replay"].get("status", "not_declared"),
    }


def write_training_recipe(path: Path, recipe: dict[str, Any]) -> None:
    validate_training_recipe(recipe)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(recipe, handle, indent=2, sort_keys=True)
        handle.write("\n")


def attach_recipe_summary(
    training_plan: dict[str, Any],
    recipe: dict[str, Any],
    recipe_path: Path,
) -> dict[str, Any]:
    updated = dict(training_plan)
    updated["training_recipe"] = {
        "status": "written",
        "path": str(recipe_path),
        "summary": training_recipe_summary(recipe),
    }
    return updated


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
        _require_non_empty_string(report, field_name)
    for field_name in ("passed", "constraints_passed", "quality_metrics_considered"):
        if not isinstance(report.get(field_name), bool):
            raise ValueError(f"{field_name} must be a bool")
    if report.get("quality_metric_policy") != QUALITY_POLICY:
        raise ValueError("quality_metric_policy must match the constraint-first policy")
    _validate_checks(report.get("constraints"), "constraints")
    _validate_checks(report.get("quality_checks"), "quality_checks")
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
        run_id=str(report.get("run_id") or report.get("attempt", {}).get("path") or report.get("cycle", "answer")),
        subject_kind="self_improvement_report",
        constraints=constraints,
        quality_checks=quality_checks,
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
            == "closed_world_lm.answer_model corpus-derived AnswerExample lessons",
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
        run_id=str(metrics.get("run_id") or metrics.get("experiment_intent_path") or "transformer"),
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


def _require_non_empty_string(record: dict[str, Any], field_name: str) -> None:
    value = record.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_string_list(record: dict[str, Any], field_name: str) -> None:
    values = record.get(field_name)
    if not isinstance(values, list) or any(
        not isinstance(value, str) or not value.strip()
        for value in values
    ):
        raise ValueError(f"{field_name} must contain only strings")


def _validate_named_rules(values: Any, field_name: str) -> None:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{field_name} must be a non-empty list")
    for value in values:
        if not isinstance(value, dict):
            raise ValueError(f"{field_name} must contain dicts")
        for key in ("name", "rule"):
            if not isinstance(value.get(key), str) or not value[key].strip():
                raise ValueError(f"{field_name} entries need {key}")
        if not isinstance(value.get("required"), bool):
            raise ValueError(f"{field_name} entries need required bool")


def _validate_checks(values: Any, field_name: str) -> None:
    if not isinstance(values, list):
        raise ValueError(f"{field_name} must be a list")
    for value in values:
        if not isinstance(value, dict):
            raise ValueError(f"{field_name} must contain dicts")
        for key in ("name", "status", "rule"):
            if not isinstance(value.get(key), str) or not value[key].strip():
                raise ValueError(f"{field_name} entries need {key}")
        if not isinstance(value.get("passed"), bool):
            raise ValueError(f"{field_name} entries need passed bool")
        if value["status"] != ("passed" if value["passed"] else "failed"):
            raise ValueError(f"{field_name} status must match passed")
        if not isinstance(value.get("details"), dict):
            raise ValueError(f"{field_name} entries need details dict")
