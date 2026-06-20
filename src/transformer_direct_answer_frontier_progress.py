"""Frontier progress guards for direct-answer snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from transformer_branch_frontier_comparison import (
    compare_metrics_to_branch_frontier,
    load_frontier_metrics,
)


def build_frontier_progress_guard(
    *,
    frontier_metrics_path: Path | str | None,
    baseline_snapshot: dict[str, Any],
    final_snapshot: dict[str, Any],
) -> dict[str, Any]:
    if frontier_metrics_path is None:
        return _inactive_guard("not_declared")

    frontier_metrics = load_frontier_metrics(Path(frontier_metrics_path))
    baseline = _compare_snapshot(baseline_snapshot, frontier_metrics)
    final = _compare_snapshot(final_snapshot, frontier_metrics)
    progress = _frontier_progress_preserved(baseline, final)
    return {
        "active": True,
        "used_for_training": False,
        "metrics_path": str(frontier_metrics_path),
        "frontier_run_id": frontier_metrics.get("run_id"),
        "baseline_comparison": baseline,
        "final_comparison": final,
        "score_non_regressed": progress["score_non_regressed"],
        "coverage_regression_count_non_increased": progress[
            "coverage_regression_count_non_increased"
        ],
        "stability_regression_count_non_increased": progress[
            "stability_regression_count_non_increased"
        ],
        "progress_preserved": progress["progress_preserved"],
        "reason": progress["reason"],
    }


def _inactive_guard(reason: str) -> dict[str, Any]:
    return {
        "active": False,
        "used_for_training": False,
        "metrics_path": None,
        "frontier_run_id": None,
        "baseline_comparison": None,
        "final_comparison": None,
        "score_non_regressed": None,
        "coverage_regression_count_non_increased": None,
        "stability_regression_count_non_increased": None,
        "progress_preserved": True,
        "reason": reason,
    }


def _frontier_progress_preserved(
    baseline: dict[str, Any] | None,
    final: dict[str, Any] | None,
) -> dict[str, Any]:
    if not _comparison_available(baseline) or not _comparison_available(final):
        return _progress_result(False, None, None, None, "comparison_unavailable")
    if final.get("passed") is True:
        return _progress_result(True, True, True, True, "frontier_passed")

    score_non_regressed = tuple(final.get("snapshot_score", ())) >= tuple(
        baseline.get("snapshot_score", ())
    )
    coverage_non_increased = _coverage_regression_count(
        final
    ) <= _coverage_regression_count(baseline)
    stability_non_increased = _stability_regression_count(
        final
    ) <= _stability_regression_count(baseline)
    preserved = (
        score_non_regressed and coverage_non_increased and stability_non_increased
    )
    reason = "progress_preserved" if preserved else "frontier_progress_regressed"
    return _progress_result(
        preserved,
        score_non_regressed,
        coverage_non_increased,
        stability_non_increased,
        reason,
    )


def _progress_result(
    progress_preserved: bool,
    score_non_regressed: bool | None,
    coverage_non_increased: bool | None,
    stability_non_increased: bool | None,
    reason: str,
) -> dict[str, Any]:
    return {
        "progress_preserved": progress_preserved,
        "score_non_regressed": score_non_regressed,
        "coverage_regression_count_non_increased": coverage_non_increased,
        "stability_regression_count_non_increased": stability_non_increased,
        "reason": reason,
    }


def _comparison_available(comparison: dict[str, Any] | None) -> bool:
    return isinstance(comparison, dict) and comparison.get("available") is True


def _coverage_regression_count(comparison: dict[str, Any]) -> int:
    diagnostics = comparison.get("coverage_diagnostics", {})
    if not isinstance(diagnostics, dict):
        return 0
    return int(diagnostics.get("violating_profile_count", 0))


def _stability_regression_count(comparison: dict[str, Any]) -> int:
    diagnostics = comparison.get("stability_diagnostics", {})
    if not isinstance(diagnostics, dict):
        return 0
    return int(diagnostics.get("violating_profile_count", 0))


def _compare_snapshot(
    snapshot: dict[str, Any],
    frontier_metrics: dict[str, Any],
) -> dict[str, Any] | None:
    return compare_metrics_to_branch_frontier(
        metrics={"direct_answer": {"final": snapshot}},
        frontier_metrics=frontier_metrics,
    )
