"""Guard-counter accounting for branch-diversity recovery attempts."""

from __future__ import annotations

from typing import Any


def record_branch_diversity_recovery_candidate(update_guard: dict[str, Any]) -> None:
    update_guard["profile_scale_branch_diversity_recovery_frontier_candidates"] += 1


def record_branch_diversity_recovery_attempt(
    update_guard: dict[str, Any],
    records: int,
) -> None:
    update_guard["profile_scale_branch_diversity_recovery_frontier_attempts"] += 1
    update_guard["profile_scale_branch_diversity_recovery_frontier_records"] += records


def record_branch_diversity_recovery_acceptance(update_guard: dict[str, Any]) -> None:
    update_guard["profile_scale_branch_diversity_recovery_frontier_acceptances"] += 1


def record_branch_diversity_recovery_fallback(update_guard: dict[str, Any]) -> None:
    update_guard[
        "profile_scale_branch_diversity_recovery_frontier_fallback_acceptances"
    ] += 1
