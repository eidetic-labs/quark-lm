"""Compact frontier-reference evidence for answer-sweep reports."""

from __future__ import annotations

from typing import Any


def frontier_reference_evidence(metrics: dict[str, Any]) -> dict[str, Any] | None:
    direct_answer = _as_dict(metrics.get("direct_answer"))
    reference = _as_dict(direct_answer.get("direct_answer_frontier_reference"))
    if not reference:
        return None
    return {
        "active": reference.get("active") is True,
        "used_for_training": reference.get("used_for_training") is True,
        "metrics_path": reference.get("metrics_path"),
        "frontier_run_id": reference.get("frontier_run_id"),
        "baseline_comparison": frontier_reference_comparison_summary(
            reference.get("baseline_comparison")
        ),
        "final_comparison": frontier_reference_comparison_summary(
            reference.get("final_comparison")
        ),
    }


def frontier_reference_active_count(trials: list[dict[str, Any]]) -> int:
    return sum(
        1
        for trial in trials
        if trial.get("direct_answer_frontier_reference", {}).get("active") is True
    )


def frontier_reference_training_use_count(trials: list[dict[str, Any]]) -> int:
    return sum(
        1
        for trial in trials
        if trial.get("direct_answer_frontier_reference", {}).get(
            "used_for_training"
        )
        is True
    )


def frontier_reference_comparison_summary(value: Any) -> dict[str, Any] | None:
    comparison = _as_dict(value)
    if not comparison:
        return None
    stability = _as_dict(comparison.get("stability_diagnostics"))
    return {
        "available": comparison.get("available") is True,
        "passed": comparison.get("passed") is True,
        "coverage_preserved": comparison.get("coverage_preserved") is True,
        "stability_preserved": comparison.get("stability_preserved") is True,
        "score_preserved": comparison.get("score_preserved") is True,
        "stability_violating_profile_count": stability.get(
            "violating_profile_count"
        ),
        "stability_worst_violation": stability.get("worst_violation"),
    }


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
