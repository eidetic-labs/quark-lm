"""Guard counter and mapping helpers for baseline-floor accounting."""

from __future__ import annotations

from typing import Any


def increment(update_guard: dict[str, Any], key: str) -> None:
    update_guard[key] += 1


def increment_map(update_guard: dict[str, Any], key: str, item: str) -> None:
    counts = update_guard[key]
    if isinstance(counts, dict):
        counts[item] = int(counts.get(item, 0)) + 1


def set_map(update_guard: dict[str, Any], key: str, item: str, value: Any) -> None:
    mapping = update_guard[key]
    if isinstance(mapping, dict):
        mapping[item] = value
