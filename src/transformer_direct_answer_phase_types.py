"""Shared result types for the direct-answer phase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DirectAnswerLoopResult:
    last_snapshot: dict[str, Any]
    last_snapshot_step: int
    routing_repair_batch_evidence: dict[str, Any] | None = None


@dataclass(frozen=True)
class DirectAnswerPhaseResult:
    model: Any
    tokenizer: Any
    optimizer: Any
    last_snapshot: dict[str, Any]
    post_direct_candidate_snapshot: dict[str, Any] | None
    post_direct_candidate_snapshot_skipped: bool
    metrics: dict[str, Any]


@dataclass(frozen=True)
class DirectAnswerPhaseRuntime:
    snapshot_recorder: Any
    baseline: dict[str, Any]
    best_snapshot: Any
    branch_context_gate: dict[str, Any]
    training_skipped: bool
    skip_reason: str | None
    steps_to_run: int
    training_cursor: Any
    params: Any
    update_guard: dict[str, Any]
    last_snapshot: dict[str, Any]
    last_snapshot_step: int
