"""Guard-counter accounting for collapsed-profile binding attempts."""

from __future__ import annotations

from typing import Any


def record_collapsed_profile_binding_candidate(update_guard: dict[str, Any]) -> None:
    update_guard["profile_scale_collapsed_profile_binding_frontier_candidates"] += 1


def record_collapsed_profile_binding_attempt(
    update_guard: dict[str, Any],
    records: int,
) -> None:
    update_guard["profile_scale_collapsed_profile_binding_frontier_attempts"] += 1
    update_guard["profile_scale_collapsed_profile_binding_frontier_records"] += records


def record_collapsed_profile_binding_acceptance(update_guard: dict[str, Any]) -> None:
    update_guard["profile_scale_collapsed_profile_binding_frontier_acceptances"] += 1


def record_collapsed_profile_binding_fallback(update_guard: dict[str, Any]) -> None:
    update_guard[
        "profile_scale_collapsed_profile_binding_frontier_fallback_acceptances"
    ] += 1
