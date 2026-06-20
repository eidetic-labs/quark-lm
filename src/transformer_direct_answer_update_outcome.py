"""Direct-answer guarded weight-update outcome summaries."""

from __future__ import annotations

from typing import Any


def direct_answer_weight_update_outcome(
    *,
    direct_steps_to_run: int,
    training_skipped: bool,
    skip_reason: str | None,
    restored_best_branch_snapshot: bool,
    restored_frontier_progress_snapshot: bool,
    frontier_progress_guard: dict[str, Any],
    update_guard: dict[str, Any] | None = None,
    require_branch_response_for_acceptance: bool = False,
) -> dict[str, Any]:
    """Summarize whether direct-answer weights were accepted or restored."""

    guard_summary = _guard_summary(update_guard)
    if training_skipped:
        return _outcome(
            status="skipped",
            accepted=False,
            reason=skip_reason or "training_skipped",
            direct_steps_to_run=direct_steps_to_run,
            restored_best_branch_snapshot=restored_best_branch_snapshot,
            restored_frontier_progress_snapshot=restored_frontier_progress_snapshot,
            frontier_progress_guard=frontier_progress_guard,
            guard_summary=guard_summary,
        )
    if direct_steps_to_run <= 0:
        return _outcome(
            status="not_run",
            accepted=True,
            reason="no_direct_answer_steps",
            direct_steps_to_run=direct_steps_to_run,
            restored_best_branch_snapshot=restored_best_branch_snapshot,
            restored_frontier_progress_snapshot=restored_frontier_progress_snapshot,
            frontier_progress_guard=frontier_progress_guard,
            guard_summary=guard_summary,
        )
    if restored_frontier_progress_snapshot:
        pre_restore = _as_dict(frontier_progress_guard.get("pre_restore"))
        return _outcome(
            status="rejected_frontier_restore",
            accepted=False,
            reason=str(pre_restore.get("reason", "frontier_progress_regressed")),
            direct_steps_to_run=direct_steps_to_run,
            restored_best_branch_snapshot=restored_best_branch_snapshot,
            restored_frontier_progress_snapshot=restored_frontier_progress_snapshot,
            frontier_progress_guard=frontier_progress_guard,
            pre_restore_progress_preserved=pre_restore.get("progress_preserved"),
            guard_summary=guard_summary,
        )
    if guard_summary["active"] and guard_summary["accepted_steps"] <= 0:
        return _outcome(
            status="rejected_guard_updates",
            accepted=False,
            reason="all_guarded_updates_rejected",
            direct_steps_to_run=direct_steps_to_run,
            restored_best_branch_snapshot=restored_best_branch_snapshot,
            restored_frontier_progress_snapshot=restored_frontier_progress_snapshot,
            frontier_progress_guard=frontier_progress_guard,
            guard_summary=guard_summary,
        )
    if (
        require_branch_response_for_acceptance
        and guard_summary["active"]
        and guard_summary["routing_repair_branch_response_acceptances"] <= 0
    ):
        return _outcome(
            status="rejected_no_branch_response",
            accepted=False,
            reason="branch_response_required",
            direct_steps_to_run=direct_steps_to_run,
            restored_best_branch_snapshot=restored_best_branch_snapshot,
            restored_frontier_progress_snapshot=restored_frontier_progress_snapshot,
            frontier_progress_guard=frontier_progress_guard,
            guard_summary=guard_summary,
        )
    if restored_best_branch_snapshot:
        return _outcome(
            status="accepted_best_snapshot_restore",
            accepted=True,
            reason="best_branch_snapshot_restored",
            direct_steps_to_run=direct_steps_to_run,
            restored_best_branch_snapshot=restored_best_branch_snapshot,
            restored_frontier_progress_snapshot=restored_frontier_progress_snapshot,
            frontier_progress_guard=frontier_progress_guard,
            guard_summary=guard_summary,
        )
    return _outcome(
        status="accepted",
        accepted=True,
        reason=_accepted_reason(guard_summary),
        direct_steps_to_run=direct_steps_to_run,
        restored_best_branch_snapshot=restored_best_branch_snapshot,
        restored_frontier_progress_snapshot=restored_frontier_progress_snapshot,
        frontier_progress_guard=frontier_progress_guard,
        guard_summary=guard_summary,
    )


def _outcome(
    *,
    status: str,
    accepted: bool,
    reason: str,
    direct_steps_to_run: int,
    restored_best_branch_snapshot: bool,
    restored_frontier_progress_snapshot: bool,
    frontier_progress_guard: dict[str, Any],
    pre_restore_progress_preserved: Any = None,
    guard_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    outcome = {
        "status": status,
        "accepted": accepted,
        "reason": reason,
        "direct_steps_to_run": direct_steps_to_run,
        "restored_best_branch_snapshot": restored_best_branch_snapshot,
        "restored_frontier_progress_snapshot": (
            restored_frontier_progress_snapshot
        ),
        "frontier_guard_active": (
            frontier_progress_guard.get("active") is True
        ),
        "frontier_progress_preserved": frontier_progress_guard.get(
            "progress_preserved"
        ),
    }
    if pre_restore_progress_preserved is not None:
        outcome["pre_restore_progress_preserved"] = pre_restore_progress_preserved
    if guard_summary is not None:
        outcome["guard"] = guard_summary
    return outcome


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _guard_summary(update_guard: dict[str, Any] | None) -> dict[str, Any]:
    guard = _as_dict(update_guard)
    attempted = int(guard.get("attempted_updates", 0))
    accepted = int(guard.get("accepted_steps", 0))
    rejected = int(guard.get("rejected_steps", 0))
    return {
        "active": attempted > 0,
        "attempted_updates": attempted,
        "accepted_steps": accepted,
        "rejected_steps": rejected,
        "routing_repair_branch_response_acceptances": int(
            guard.get("routing_repair_branch_response_acceptances", 0)
        ),
        "routing_repair_neutral_update_acceptances": int(
            guard.get("routing_repair_neutral_update_acceptances", 0)
        ),
        "frontier_update_guard_active": (
            guard.get("frontier_update_guard_active") is True
        ),
    }


def _accepted_reason(guard_summary: dict[str, Any]) -> str:
    if guard_summary["active"]:
        branch_acceptances = guard_summary[
            "routing_repair_branch_response_acceptances"
        ]
        neutral_acceptances = guard_summary[
            "routing_repair_neutral_update_acceptances"
        ]
        if neutral_acceptances > 0 and branch_acceptances <= 0:
            return "guarded_neutral_updates_accepted"
        return "guarded_updates_accepted"
    return "frontier_progress_preserved"
