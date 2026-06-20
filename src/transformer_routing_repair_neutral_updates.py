"""Neutral safe-update accounting for routing-repair search."""

from __future__ import annotations

from typing import Any


ROUTING_REPAIR_NEUTRAL_UPDATE_SHAPE = "routing_repair_neutral"


def record_routing_repair_neutral_acceptance(
    guard: dict[str, Any],
    learning_rate_scale: float,
) -> None:
    """Record a safe routing-repair update that has no immediate branch response."""

    guard["routing_repair_neutral_update_acceptances"] = int(
        guard.get("routing_repair_neutral_update_acceptances", 0)
    ) + 1
    scales = guard.setdefault("routing_repair_neutral_learning_rate_scales", [])
    if isinstance(scales, list):
        scales.append(learning_rate_scale)
