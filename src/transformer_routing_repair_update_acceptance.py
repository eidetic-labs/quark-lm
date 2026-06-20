"""Acceptance helpers for routing-repair update search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformer_direct_answer_mode_dispatch import DirectAnswerModeStepResult
from transformer_direct_answer_update_guard import (
    record_direct_update_guard_acceptance,
)


@dataclass(frozen=True)
class RoutingRepairNeutralCandidate:
    learning_rate_scale: float
    loss: float


def accept_branch_response_update(
    guard: dict[str, Any],
    learning_rate_scale: float,
    loss: float,
    update_shape: str,
) -> DirectAnswerModeStepResult:
    record_branch_response(guard, learning_rate_scale)
    record_direct_update_guard_acceptance(
        guard,
        learning_rate_scale,
        update_shape,
    )
    guard["routing_repair_accepted_learning_rate_scale"] = learning_rate_scale
    return DirectAnswerModeStepResult(loss, update_guard_applied=True)


def record_branch_response(
    guard: dict[str, Any],
    learning_rate_scale: float,
) -> None:
    guard["routing_repair_branch_response_acceptances"] = int(
        guard.get("routing_repair_branch_response_acceptances", 0)
    ) + 1
    scales = guard.setdefault("routing_repair_branch_response_learning_rate_scales", [])
    if isinstance(scales, list):
        scales.append(learning_rate_scale)
