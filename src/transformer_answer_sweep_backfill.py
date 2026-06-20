"""Rebuild answer-sweep reports from existing trial metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformer_answer_sweep_report import (
    build_answer_sweep_report,
    trial_report_from_metrics,
)


def rebuild_answer_sweep_report_from_metrics(
    *,
    source_report_path: Path,
    frontier_metrics: dict[str, Any] | None,
    frontier_metrics_path: Path | None,
) -> dict[str, Any]:
    source_report = _load_report(source_report_path)
    rebuilt_trials = [
        _trial_report_from_existing_metrics(
            trial=trial,
            frontier_metrics=frontier_metrics,
        )
        for trial in source_report.get("trials", [])
    ]
    report = build_answer_sweep_report(
        run_id=str(source_report.get("run_id", source_report_path.parent.name)),
        axes=_as_dict(source_report.get("axes")),
        trials=rebuilt_trials,
        max_trials=int(source_report.get("max_trials", len(rebuilt_trials))),
        dry_run=False,
        frontier_metrics_path=frontier_metrics_path,
    )
    report["source_report_path"] = str(source_report_path)
    report["report_mode"] = "existing_metrics_backfill"
    return report


def _trial_report_from_existing_metrics(
    *,
    trial: dict[str, Any],
    frontier_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    run_path = Path(str(trial.get("run", "")))
    metrics_path = _trial_metrics_path(trial, run_path)
    metrics = _load_report(metrics_path)
    metrics["metrics_path"] = str(metrics_path)
    return trial_report_from_metrics(
        trial_id=str(trial.get("trial_id", run_path.name)),
        run_path=run_path,
        config=_as_dict(trial.get("config")),
        metrics=metrics,
        frontier_metrics=frontier_metrics,
    )


def _trial_metrics_path(trial: dict[str, Any], run_path: Path) -> Path:
    metrics_path = trial.get("metrics_path")
    if isinstance(metrics_path, str) and metrics_path:
        return Path(metrics_path)
    if not str(run_path):
        raise ValueError("existing sweep trial is missing a run path")
    return run_path / "transformer_answer_metrics.json"


def _load_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"expected JSON object at {path}")
    return loaded


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
