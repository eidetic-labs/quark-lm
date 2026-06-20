"""Constraint checks for guarded direct-answer weight updates."""

from __future__ import annotations

from typing import Any

from constraint_first_report import promotion_check


UPDATE_OUTCOME_CONSTRAINT_NAME = "guarded_direct_answer_update_outcome"
UPDATE_OUTCOME_RULE = (
    "Attempted direct-answer weight updates must be accepted by the guard "
    "or explicitly not run before promotion."
)
ACCEPTED_OUTCOME_STATUSES = {
    "accepted",
    "accepted_best_snapshot_restore",
    "not_run",
}


def direct_answer_update_outcome_check(direct_answer: Any) -> dict[str, Any]:
    """Build the promotion constraint for guarded update outcomes."""

    if not isinstance(direct_answer, dict):
        return _failed_check("direct_answer evidence is missing")
    outcome = direct_answer.get("direct_answer_weight_update_outcome")
    if not isinstance(outcome, dict):
        return _failed_check("direct-answer update outcome is missing")
    status = outcome.get("status")
    accepted = outcome.get("accepted") is True
    passed = accepted and status in ACCEPTED_OUTCOME_STATUSES
    return promotion_check(
        UPDATE_OUTCOME_CONSTRAINT_NAME,
        passed,
        UPDATE_OUTCOME_RULE,
        {
            "status": status,
            "accepted": outcome.get("accepted"),
            "reason": outcome.get("reason"),
            "direct_steps_to_run": outcome.get("direct_steps_to_run"),
            "restored_best_branch_snapshot": outcome.get(
                "restored_best_branch_snapshot"
            ),
            "restored_frontier_progress_snapshot": outcome.get(
                "restored_frontier_progress_snapshot"
            ),
        },
    )


def _failed_check(error: str) -> dict[str, Any]:
    return promotion_check(
        UPDATE_OUTCOME_CONSTRAINT_NAME,
        False,
        UPDATE_OUTCOME_RULE,
        {"error": error},
    )
