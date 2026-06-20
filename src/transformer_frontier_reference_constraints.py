"""Promotion constraints for direct-answer frontier references."""

from __future__ import annotations

from typing import Any

from constraint_first_report import promotion_check


REFERENCE_KEY = "direct_answer_frontier_reference"


def direct_answer_frontier_reference_checks(
    direct_answer: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    evidence = _frontier_reference_evidence(direct_answer)
    return [
        promotion_check(
            "direct_answer_frontier_reference_purity",
            evidence["purity_passed"],
            (
                "Frontier reference metrics may be used only as evaluation evidence, "
                "never as training data."
            ),
            evidence,
        ),
        promotion_check(
            "direct_answer_frontier_reference_final",
            evidence["final_passed"],
            (
                "Active frontier references require final direct-answer evidence to "
                "preserve coverage, stability, and score against the frontier."
            ),
            evidence,
        ),
    ]


def _frontier_reference_evidence(
    direct_answer: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(direct_answer, dict) or REFERENCE_KEY not in direct_answer:
        return _inactive_evidence("not_recorded")

    reference = direct_answer.get(REFERENCE_KEY)
    if not isinstance(reference, dict):
        return {
            "active": None,
            "available": False,
            "reason": "malformed_reference",
            "used_for_training": None,
            "metrics_path": None,
            "frontier_run_id": None,
            "final_comparison": None,
            "purity_passed": False,
            "final_passed": False,
        }

    active = reference.get("active") is True
    used_for_training = reference.get("used_for_training")
    final_comparison = reference.get("final_comparison")
    return {
        "active": active,
        "available": True,
        "reason": "active_reference" if active else "inactive_reference",
        "used_for_training": used_for_training,
        "metrics_path": reference.get("metrics_path"),
        "frontier_run_id": reference.get("frontier_run_id"),
        "final_comparison": final_comparison,
        "purity_passed": _purity_passed(active, used_for_training),
        "final_passed": _final_comparison_passed(active, final_comparison),
    }


def _inactive_evidence(reason: str) -> dict[str, Any]:
    return {
        "active": False,
        "available": False,
        "reason": reason,
        "used_for_training": None,
        "metrics_path": None,
        "frontier_run_id": None,
        "final_comparison": None,
        "purity_passed": True,
        "final_passed": True,
    }


def _purity_passed(active: bool, used_for_training: Any) -> bool:
    if active:
        return used_for_training is False
    return used_for_training is not True


def _final_comparison_passed(active: bool, comparison: Any) -> bool:
    if not active:
        return True
    return isinstance(comparison, dict) and comparison.get("passed") is True
