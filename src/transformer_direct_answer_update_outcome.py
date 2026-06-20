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
) -> dict[str, Any]:
    """Summarize whether direct-answer weights were accepted or restored."""

    if training_skipped:
        return _outcome(
            status="skipped",
            accepted=False,
            reason=skip_reason or "training_skipped",
            direct_steps_to_run=direct_steps_to_run,
            restored_best_branch_snapshot=restored_best_branch_snapshot,
            restored_frontier_progress_snapshot=restored_frontier_progress_snapshot,
            frontier_progress_guard=frontier_progress_guard,
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
        )
    return _outcome(
        status="accepted",
        accepted=True,
        reason="frontier_progress_preserved",
        direct_steps_to_run=direct_steps_to_run,
        restored_best_branch_snapshot=restored_best_branch_snapshot,
        restored_frontier_progress_snapshot=restored_frontier_progress_snapshot,
        frontier_progress_guard=frontier_progress_guard,
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
    return outcome


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
