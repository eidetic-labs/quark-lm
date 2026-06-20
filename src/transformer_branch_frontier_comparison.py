"""Frontier comparisons for transformer branch-diversity evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from branch_diversity_snapshot_coverage import (
    branch_diversity_snapshot_target_coverage_delta,
    branch_diversity_snapshot_target_coverage_diagnostics,
)
from branch_diversity_snapshots import branch_diversity_snapshot_score
from transformer_branch_frontier_profile_diagnostics import (
    branch_frontier_profile_regression_diagnostics,
)


def load_frontier_metrics(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"frontier metrics must be a JSON object: {path}")
    loaded["metrics_path"] = str(path)
    return loaded


def compare_metrics_to_branch_frontier(
    *,
    metrics: dict[str, Any],
    frontier_metrics: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if frontier_metrics is None:
        return None
    snapshot = _direct_answer_final_snapshot(metrics)
    frontier_snapshot = _direct_answer_final_snapshot(frontier_metrics)
    if not snapshot or not frontier_snapshot:
        return {
            "available": False,
            "passed": False,
            "reason": "missing_direct_answer_final_snapshot",
        }

    coverage_diagnostics = branch_diversity_snapshot_target_coverage_diagnostics(
        snapshot,
        frontier_snapshot,
    )
    coverage_delta = branch_diversity_snapshot_target_coverage_delta(
        snapshot,
        frontier_snapshot,
    )
    snapshot_score = branch_diversity_snapshot_score(snapshot)
    frontier_score = branch_diversity_snapshot_score(frontier_snapshot)
    score_direction = _score_direction(snapshot_score, frontier_score)
    coverage_preserved = bool(coverage_diagnostics["preserved"])
    score_preserved = score_direction >= 0
    return {
        "available": True,
        "passed": coverage_preserved and score_preserved,
        "frontier_run_id": frontier_metrics.get("run_id"),
        "frontier_metrics_path": frontier_metrics.get("metrics_path"),
        "coverage_preserved": coverage_preserved,
        "score_preserved": score_preserved,
        "score_direction": score_direction,
        "snapshot_score": list(snapshot_score),
        "frontier_score": list(frontier_score),
        "coverage_diagnostics": coverage_diagnostics,
        "coverage_delta": coverage_delta,
        "snapshot_branch_diversity_passed": _branch_diversity_passed(snapshot),
        "frontier_branch_diversity_passed": _branch_diversity_passed(
            frontier_snapshot
        ),
        "profile_regression_diagnostics": (
            branch_frontier_profile_regression_diagnostics(
                snapshot=snapshot,
                frontier_snapshot=frontier_snapshot,
            )
        ),
    }


def _direct_answer_final_snapshot(metrics: dict[str, Any]) -> dict[str, Any]:
    direct_answer = metrics.get("direct_answer")
    if not isinstance(direct_answer, dict):
        return {}
    final = direct_answer.get("final")
    return final if isinstance(final, dict) else {}


def _branch_diversity_passed(snapshot: dict[str, Any]) -> bool | None:
    target = snapshot.get("branch_diversity_target")
    if not isinstance(target, dict):
        return None
    passed = target.get("passed")
    return passed if isinstance(passed, bool) else None


def _score_direction(
    snapshot_score: tuple[float, ...],
    frontier_score: tuple[float, ...],
) -> int:
    if snapshot_score > frontier_score:
        return 1
    if snapshot_score < frontier_score:
        return -1
    return 0
