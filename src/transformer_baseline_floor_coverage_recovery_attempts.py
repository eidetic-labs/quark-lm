"""Attempt state for baseline-floor coverage recovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformer_baseline_floor_profile_outcome_types import (
    BaselineFloorCoverageRecoveryOutcome,
)


@dataclass
class CoverageRecoveryAttemptState:
    loss_total: float = 0.0
    loss_count: int = 0
    attempted: bool = False
    recovery_outcome: str = "not_attempted"
    rejection_reason: str = ""
    recovery_delta: dict[str, Any] | None = None
    recovery_score: tuple[float, ...] | None = None
    branch_stable_checked: bool = False
    branch_stable_accepted: bool = False
    branch_stability_preserved: bool | None = None

    def record_loss(self, loss: float) -> None:
        self.attempted = True
        self.loss_total += loss
        self.loss_count += 1

    def record_branch_stable_check(self) -> None:
        self.branch_stable_checked = True

    def apply_outcome(
        self,
        outcome: BaselineFloorCoverageRecoveryOutcome,
        *,
        recovery_delta: dict[str, Any],
        recovery_score: tuple[float, ...],
    ) -> None:
        self.recovery_delta = recovery_delta
        self.recovery_score = recovery_score
        self.recovery_outcome = outcome.outcome
        self.rejection_reason = outcome.rejection_reason
        if outcome.branch_stability_preserved is not None:
            self.branch_stability_preserved = outcome.branch_stability_preserved
        if outcome.branch_stable_accepted:
            self.branch_stable_accepted = True

    def mark_coverage_tie(self) -> None:
        self.recovery_outcome = "coverage_tied"
        self.rejection_reason = "coverage_tie"
