"""Frontier-progress guards for direct-answer weight updates."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from transformer_direct_answer_frontier_progress import build_frontier_progress_guard
from transformer_direct_answer_update_guard import (
    record_direct_update_guard_acceptance,
)
from transformer_direct_answer_update_rejections import (
    record_direct_update_guard_rejection_attempt,
)

FRONTIER_UPDATE_SHAPE = "frontier_progress"


def direct_answer_frontier_update_guard_active(args: Any) -> bool:
    return getattr(args, "direct_answer_frontier_metrics", None) is not None


def apply_direct_frontier_update_guard_probe(
    *,
    direct_answer_update_guard: dict[str, Any],
    direct_baseline: dict[str, Any],
    direct_step: int,
    direct_snapshot_recorder: Any,
    frontier_metrics_path: Any | None,
    pre_update_model_payload: dict[str, Any] | None,
    pre_update_optimizer_payload: dict[str, Any] | None,
    restore_direct_update_state: Callable[[dict[str, Any], dict[str, Any]], None],
) -> bool:
    """Restore a direct update when it regresses declared frontier progress."""

    direct_answer_update_guard["frontier_update_guard_active"] = True
    direct_answer_update_guard["checked_steps"] += 1
    direct_answer_update_guard["attempted_updates"] += 1
    probe_snapshot = direct_snapshot_recorder.record(
        direct_step,
        None,
        {
            "frontier_update_guard_probe": True,
            "learning_rate_scale": 1.0,
        },
    )
    frontier_guard = build_frontier_progress_guard(
        frontier_metrics_path=frontier_metrics_path,
        baseline_snapshot=direct_baseline,
        final_snapshot=probe_snapshot,
    )
    direct_answer_update_guard["frontier_update_guard_last"] = _guard_summary(
        frontier_guard
    )
    if frontier_guard.get("progress_preserved") is True:
        _record_frontier_acceptance(direct_answer_update_guard)
        record_direct_update_guard_acceptance(
            direct_answer_update_guard,
            1.0,
            FRONTIER_UPDATE_SHAPE,
        )
        return True

    _record_frontier_rejection(direct_answer_update_guard, frontier_guard)
    direct_answer_update_guard["rejected_steps"] += 1
    record_direct_update_guard_rejection_attempt(
        direct_answer_update_guard,
        direct_baseline,
        direct_step,
        probe_snapshot,
        1.0,
        FRONTIER_UPDATE_SHAPE,
    )
    if pre_update_model_payload is not None and pre_update_optimizer_payload is not None:
        restore_direct_update_state(
            pre_update_model_payload,
            pre_update_optimizer_payload,
        )
    return False


def _record_frontier_acceptance(guard: dict[str, Any]) -> None:
    guard["frontier_update_guard_acceptances"] = int(
        guard.get("frontier_update_guard_acceptances", 0)
    ) + 1


def _record_frontier_rejection(
    guard: dict[str, Any],
    frontier_guard: dict[str, Any],
) -> None:
    guard["frontier_update_guard_rejections"] = int(
        guard.get("frontier_update_guard_rejections", 0)
    ) + 1
    samples = guard.setdefault("frontier_update_guard_rejection_sample", [])
    if isinstance(samples, list) and len(samples) < 5:
        samples.append(_guard_summary(frontier_guard))


def _guard_summary(frontier_guard: dict[str, Any]) -> dict[str, Any]:
    return {
        "active": frontier_guard.get("active") is True,
        "reason": frontier_guard.get("reason"),
        "progress_preserved": frontier_guard.get("progress_preserved"),
        "score_non_regressed": frontier_guard.get("score_non_regressed"),
        "coverage_regression_count_non_increased": frontier_guard.get(
            "coverage_regression_count_non_increased"
        ),
        "stability_regression_count_non_increased": frontier_guard.get(
            "stability_regression_count_non_increased"
        ),
    }
