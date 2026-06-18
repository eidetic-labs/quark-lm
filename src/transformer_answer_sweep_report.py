"""Report assembly for controlled answer-training sweeps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SWEEP_REPORT_KIND = "transformer_answer_sweep_report"
SWEEP_REPORT_SCHEMA_VERSION = 1


def build_answer_sweep_report(
    *,
    run_id: str,
    axes: dict[str, list[Any]],
    trials: list[dict[str, Any]],
    max_trials: int,
    dry_run: bool,
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
) -> dict[str, Any]:
    promotion = metrics.get("constraint_first_promotion", {})
    return {
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
    }


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
    }
