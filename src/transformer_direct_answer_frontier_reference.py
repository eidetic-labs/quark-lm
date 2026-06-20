"""Direct-answer frontier reference evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from transformer_branch_frontier_comparison import (
    compare_metrics_to_branch_frontier,
    load_frontier_metrics,
)


def build_direct_answer_frontier_reference(
    *,
    args: Any,
    direct_baseline: dict[str, Any],
    final_snapshot: dict[str, Any],
) -> dict[str, Any]:
    path = getattr(args, "direct_answer_frontier_metrics", None)
    if path is None:
        return _inactive_reference()
    frontier_metrics = load_frontier_metrics(Path(path))
    return {
        "active": True,
        "used_for_training": False,
        "metrics_path": str(path),
        "frontier_run_id": frontier_metrics.get("run_id"),
        "rule": (
            "Frontier metrics are an evaluation reference for guard evidence; "
            "they are not training data, weights, tokenizer state, or embeddings."
        ),
        "baseline_comparison": _compare_snapshot(direct_baseline, frontier_metrics),
        "final_comparison": _compare_snapshot(final_snapshot, frontier_metrics),
    }


def _inactive_reference() -> dict[str, Any]:
    return {
        "active": False,
        "used_for_training": False,
        "metrics_path": None,
        "frontier_run_id": None,
        "baseline_comparison": None,
        "final_comparison": None,
    }


def _compare_snapshot(
    snapshot: dict[str, Any],
    frontier_metrics: dict[str, Any],
) -> dict[str, Any] | None:
    return compare_metrics_to_branch_frontier(
        metrics={"direct_answer": {"final": snapshot}},
        frontier_metrics=frontier_metrics,
    )
