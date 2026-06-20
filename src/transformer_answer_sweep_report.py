"""Report assembly for controlled answer-training sweeps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformer_branch_frontier_comparison import (
    compare_metrics_to_branch_frontier,
)


SWEEP_REPORT_KIND = "transformer_answer_sweep_report"
SWEEP_REPORT_SCHEMA_VERSION = 1


def build_answer_sweep_report(
    *,
    run_id: str,
    axes: dict[str, list[Any]],
    trials: list[dict[str, Any]],
    max_trials: int,
    dry_run: bool,
    frontier_metrics_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SWEEP_REPORT_SCHEMA_VERSION,
        "kind": SWEEP_REPORT_KIND,
        "component": "transformer-answer-sweep",
        "run_id": run_id,
        "status": "planned" if dry_run else _completion_status(trials),
        "rule": (
            "Tokenizer, architecture, optimizer, and epoch-budget changes must "
            "be compared through declared trials instead of undocumented knobs."
        ),
        "axes": axes,
        "trial_count": len(trials),
        "max_trials": max_trials,
        "trials": trials,
        "summary": _summary(trials, dry_run),
        "frontier_metrics_path": (
            str(frontier_metrics_path) if frontier_metrics_path is not None else None
        ),
        "pretrained_weights": False,
        "pretrained_tokenizer": False,
        "external_embeddings": False,
    }


def trial_report_from_metrics(
    *,
    trial_id: str,
    run_path: Path,
    config: dict[str, Any],
    metrics: dict[str, Any],
    frontier_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    promotion = metrics.get("constraint_first_promotion", {})
    branch_evidence = _branch_evidence(metrics)
    report = {
        "trial_id": trial_id,
        "status": "completed",
        "run": str(run_path),
        "config": config,
        "metrics_path": metrics.get("metrics_path"),
        "checkpoint": metrics.get("checkpoint"),
        "tokenizer_type": metrics.get("tokenizer_type"),
        "tokenizer_manifest_hash": metrics.get("tokenizer_manifest_hash"),
        "transformer_profile": metrics.get("transformer_profile"),
        "constraint_status": promotion.get("status"),
        "failed_constraints": promotion.get("failed_constraints", []),
        "quality_metrics_considered": promotion.get("quality_metrics_considered"),
        "direct_answer_branch_evidence": branch_evidence,
    }
    frontier_comparison = compare_metrics_to_branch_frontier(
        metrics=metrics,
        frontier_metrics=frontier_metrics,
    )
    if frontier_comparison is not None:
        report["frontier_comparison"] = frontier_comparison
    return report


def planned_trial_report(
    *,
    trial_id: str,
    run_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "trial_id": trial_id,
        "status": "planned",
        "run": str(run_path),
        "config": config,
    }


def write_answer_sweep_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _completion_status(trials: list[dict[str, Any]]) -> str:
    if all(trial.get("status") == "completed" for trial in trials):
        return "completed"
    return "incomplete"


def _summary(trials: list[dict[str, Any]], dry_run: bool) -> dict[str, Any]:
    completed = [trial for trial in trials if trial.get("status") == "completed"]
    return {
        "passed": (len(completed) == len(trials)) if not dry_run else True,
        "completed_trials": len(completed),
        "planned_trials": len(trials),
        "dry_run": dry_run,
        "tokenizer_types": sorted(
            {
                trial.get("tokenizer_type")
                for trial in completed
                if trial.get("tokenizer_type") is not None
            }
        ),
        "constraint_statuses": sorted(
            {
                trial.get("constraint_status")
                for trial in completed
                if trial.get("constraint_status") is not None
            }
        ),
        "branch_diversity_passed_trials": sum(
            1
            for trial in completed
            if trial.get("direct_answer_branch_evidence", {})
            .get("branch_diversity_target", {})
            .get("passed")
            is True
        ),
        "frontier_competitive_trials": sum(
            1
            for trial in completed
            if trial.get("frontier_comparison", {}).get("passed") is True
        ),
        "frontier_regressed_trials": sum(
            1
            for trial in completed
            if trial.get("frontier_comparison", {}).get("available") is True
            and trial.get("frontier_comparison", {}).get("passed") is False
        ),
    }


def _branch_evidence(metrics: dict[str, Any]) -> dict[str, Any]:
    direct_answer = _as_dict(metrics.get("direct_answer"))
    final = _as_dict(direct_answer.get("final"))
    diversity = _as_dict(final.get("branch_diversity_target"))
    root_cause = _as_dict(diversity.get("root_cause"))
    guard = _as_dict(direct_answer.get("direct_answer_update_guard"))
    routing = _as_dict(direct_answer.get("routing_repair_batch_evidence"))
    return {
        "direct_answer_mode": direct_answer.get("direct_answer_mode"),
        "actual_steps": direct_answer.get("actual_steps"),
        "branch_target_coverage_by_profile": final.get(
            "branch_target_coverage_by_profile",
            {},
        ),
        "branch_diversity_target": {
            "passed": diversity.get("passed"),
            "failed_profiles": diversity.get("failed_profiles"),
            "passed_profiles": diversity.get("passed_profiles"),
            "max_dominant_predicted_rate": diversity.get(
                "max_dominant_predicted_rate"
            ),
            "min_target_token_coverage": diversity.get(
                "min_target_token_coverage"
            ),
            "mode_counts": root_cause.get("mode_counts", {}),
        },
        "update_guard": {
            "accepted_steps": guard.get("accepted_steps"),
            "rejected_steps": guard.get("rejected_steps"),
            "accepted_learning_rate_scale_counts": guard.get(
                "accepted_learning_rate_scale_counts",
                {},
            ),
        },
        "routing_repair_batch_evidence": {
            "passed": routing.get("passed"),
            "branch_count": routing.get("branch_count"),
            "retention_anchor_count": routing.get("retention_anchor_count"),
        },
    }


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
