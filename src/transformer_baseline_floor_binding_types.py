"""Collapsed profile binding result payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CollapsedProfileBindingResult:
    loss_total: float
    loss_count: int
    floor_preserved: bool
    profile_probe_snapshot: dict[str, Any] | None
    profile_score: tuple[float, ...] | None
    diversity_outcome: str
    diversity_rejection_reason: str
    owner_paraphrase_binding_preservation_delta: dict[str, Any] | None
    attempted: bool = False
    accepted: bool = False
    outcome: str = "not_attempted"
    rejection_reason: str = ""
    learning_rate_scale: float | None = None
    records: int = 0
    target_profiles: list[str] | None = None
    base_score: tuple[float, ...] | None = None
    score: tuple[float, ...] | None = None
    delta: dict[str, Any] | None = None

