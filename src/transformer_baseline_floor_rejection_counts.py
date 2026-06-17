"""Shared guard counter helpers for baseline-floor rejection accounting."""

from __future__ import annotations

from typing import Any


def increment_rejection_counter(update_guard: dict[str, Any], key: str) -> None:
    update_guard[key] += 1


def increment_rejection_reason(
    update_guard: dict[str, Any],
    key: str,
    reason: str,
) -> None:
    counts = update_guard[key]
    if isinstance(counts, dict):
        counts[reason] = int(counts.get(reason, 0)) + 1
