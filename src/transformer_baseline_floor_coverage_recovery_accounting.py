"""Guard accounting for baseline-floor coverage recovery."""

from __future__ import annotations

from typing import Any


def record_coverage_recovery_candidate(update_guard: dict[str, Any]) -> None:
    update_guard["profile_scale_coverage_recovery_frontier_prepared_candidates"] += 1


def record_coverage_recovery_attempt(
    update_guard: dict[str, Any],
    records: int,
) -> None:
    update_guard["profile_scale_coverage_recovery_frontier_attempts"] += 1
    update_guard["profile_scale_coverage_recovery_frontier_records"] += records


def record_branch_stable_coverage_check(update_guard: dict[str, Any]) -> None:
    update_guard["profile_scale_branch_stable_coverage_recovery_frontier_checks"] += 1


def record_coverage_recovery_acceptance(
    update_guard: dict[str, Any],
    *,
    branch_stable_active: bool,
) -> None:
    update_guard["profile_scale_coverage_recovery_frontier_acceptances"] += 1
    if branch_stable_active:
        update_guard[
            "profile_scale_branch_stable_coverage_recovery_frontier_acceptances"
        ] += 1
