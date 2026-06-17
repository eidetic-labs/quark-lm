"""Baseline-floor profile-scale outcome payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BaselineFloorProfileOutcome:
    floor_preserved: bool
    diversity_outcome: str = "not_active"
    diversity_rejection_reason: str = "floor_regression"
    profile_score: tuple[float, ...] | None = None
    coverage_outcome: str = "not_active"
    coverage_rejection_reason: str = "floor_regression"
    coverage_delta: dict[str, Any] | None = None
    coverage_prep_accepted: bool = False


@dataclass(frozen=True)
class BaselineFloorCoverageRecoveryOutcome:
    accepted: bool
    outcome: str
    rejection_reason: str
    branch_stability_preserved: bool | None = None
    branch_stable_accepted: bool = False


@dataclass(frozen=True)
class BaselineFloorBranchDiversityRecoveryOutcome:
    accepted: bool
    outcome: str
    rejection_reason: str


@dataclass(frozen=True)
class BaselineFloorCollapsedProfileBindingOutcome:
    accepted: bool
    outcome: str
    rejection_reason: str
    owner_paraphrase_preservation_failed: bool = False


@dataclass(frozen=True)
class BaselineFloorMissingFirstTokenOutcome:
    accepted: bool
    outcome: str
    rejection_reason: str

