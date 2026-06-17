"""Baseline-floor profile-scale attempt state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformer_baseline_floor_owner_preservation import (
    OwnerParaphraseBindingPreservation,
)
from transformer_baseline_floor_profile_outcome_types import BaselineFloorProfileOutcome


@dataclass
class BaselineFloorProfileAttemptState:
    floor_preserved: bool
    diversity_outcome: str
    diversity_rejection_reason: str
    profile_score: tuple[float, ...] | None
    profile_base_score: tuple[float, ...] | None
    coverage_outcome: str
    coverage_rejection_reason: str
    coverage_delta: dict[str, Any] | None
    coverage_prep_accepted: bool
    profile_probe_snapshot: dict[str, Any] | None = None
    coverage_recovery_attempted: bool = False
    coverage_recovery_accepted: bool = False
    coverage_recovery_outcome: str = "not_attempted"
    coverage_recovery_rejection_reason: str = ""
    coverage_recovery_learning_rate_scale: float | None = None
    coverage_recovery_records: int = 0
    coverage_recovery_delta: dict[str, Any] | None = None
    coverage_recovery_prepared_score: tuple[float, ...] | None = None
    coverage_recovery_score: tuple[float, ...] | None = None
    coverage_recovery_branch_stable_checked: bool = False
    coverage_recovery_branch_stable_accepted: bool = False
    coverage_recovery_branch_stability_preserved: bool | None = None
    branch_diversity_recovery_attempted: bool = False
    branch_diversity_recovery_accepted: bool = False
    branch_diversity_recovery_outcome: str = "not_attempted"
    branch_diversity_recovery_rejection_reason: str = ""
    branch_diversity_recovery_learning_rate_scale: float | None = None
    branch_diversity_recovery_records: int = 0
    branch_diversity_recovery_base_score: tuple[float, ...] | None = None
    branch_diversity_recovery_score: tuple[float, ...] | None = None
    branch_diversity_recovery_delta: dict[str, Any] | None = None
    collapsed_profile_binding_attempted: bool = False
    collapsed_profile_binding_accepted: bool = False
    collapsed_profile_binding_outcome: str = "not_attempted"
    collapsed_profile_binding_rejection_reason: str = ""
    collapsed_profile_binding_learning_rate_scale: float | None = None
    collapsed_profile_binding_records: int = 0
    collapsed_profile_binding_target_profiles: list[str] | None = None
    collapsed_profile_binding_base_score: tuple[float, ...] | None = None
    collapsed_profile_binding_score: tuple[float, ...] | None = None
    collapsed_profile_binding_delta: dict[str, Any] | None = None
    missing_first_token_attempted: bool = False
    missing_first_token_accepted: bool = False
    missing_first_token_outcome: str = "not_attempted"
    missing_first_token_rejection_reason: str = ""
    missing_first_token_learning_rate_scale: float | None = None
    missing_first_token_records: int = 0
    missing_first_token_target_profiles: list[str] | None = None
    missing_first_token_target_ids: list[int] | None = None
    missing_first_token_base_score: tuple[float, ...] | None = None
    missing_first_token_score: tuple[float, ...] | None = None
    missing_first_token_delta: dict[str, Any] | None = None
    owner_paraphrase_binding_preserved: bool = True
    owner_paraphrase_binding_preservation_delta: dict[str, Any] | None = None

    @classmethod
    def from_profile_outcome(
        cls,
        profile_outcome: BaselineFloorProfileOutcome,
        profile_probe_snapshot: dict[str, Any],
        profile_base_score: tuple[float, ...] | None,
    ) -> "BaselineFloorProfileAttemptState":
        return cls(
            floor_preserved=profile_outcome.floor_preserved,
            diversity_outcome=profile_outcome.diversity_outcome,
            diversity_rejection_reason=profile_outcome.diversity_rejection_reason,
            profile_score=profile_outcome.profile_score,
            profile_base_score=profile_base_score,
            coverage_outcome=profile_outcome.coverage_outcome,
            coverage_rejection_reason=profile_outcome.coverage_rejection_reason,
            coverage_delta=profile_outcome.coverage_delta,
            coverage_prep_accepted=profile_outcome.coverage_prep_accepted,
            profile_probe_snapshot=profile_probe_snapshot,
            coverage_recovery_prepared_score=profile_outcome.profile_score,
            collapsed_profile_binding_target_profiles=[],
            missing_first_token_target_profiles=[],
            missing_first_token_target_ids=[],
        )

    def apply_coverage_recovery(self, result: Any) -> None:
        self.floor_preserved = result.floor_preserved
        self.profile_probe_snapshot = result.profile_probe_snapshot
        self.profile_score = result.profile_score
        self.diversity_outcome = result.diversity_outcome
        self.diversity_rejection_reason = result.diversity_rejection_reason
        self.coverage_delta = result.coverage_delta
        self.coverage_outcome = result.coverage_outcome
        self.coverage_rejection_reason = result.coverage_rejection_reason
        self.coverage_prep_accepted = result.coverage_prep_accepted
        self.coverage_recovery_attempted = result.attempted
        self.coverage_recovery_accepted = result.accepted
        self.coverage_recovery_outcome = result.outcome
        self.coverage_recovery_rejection_reason = result.rejection_reason
        self.coverage_recovery_learning_rate_scale = result.learning_rate_scale
        self.coverage_recovery_records = result.records
        self.coverage_recovery_delta = result.delta
        self.coverage_recovery_prepared_score = result.prepared_score
        self.coverage_recovery_score = result.score
        self.coverage_recovery_branch_stable_checked = result.branch_stable_checked
        self.coverage_recovery_branch_stable_accepted = (
            result.branch_stable_accepted
        )
        self.coverage_recovery_branch_stability_preserved = (
            result.branch_stability_preserved
        )

    def apply_branch_diversity_recovery(self, result: Any) -> None:
        self.floor_preserved = result.floor_preserved
        self.profile_probe_snapshot = result.profile_probe_snapshot
        self.profile_score = result.profile_score
        self.diversity_outcome = result.diversity_outcome
        self.diversity_rejection_reason = result.diversity_rejection_reason
        self.branch_diversity_recovery_attempted = result.attempted
        self.branch_diversity_recovery_accepted = result.accepted
        self.branch_diversity_recovery_outcome = result.outcome
        self.branch_diversity_recovery_rejection_reason = result.rejection_reason
        self.branch_diversity_recovery_learning_rate_scale = result.learning_rate_scale
        self.branch_diversity_recovery_records = result.records
        self.branch_diversity_recovery_base_score = result.base_score
        self.branch_diversity_recovery_score = result.score
        self.branch_diversity_recovery_delta = result.delta

    def apply_collapsed_profile_binding(self, result: Any) -> None:
        self.floor_preserved = result.floor_preserved
        self.profile_probe_snapshot = result.profile_probe_snapshot
        self.profile_score = result.profile_score
        self.diversity_outcome = result.diversity_outcome
        self.diversity_rejection_reason = result.diversity_rejection_reason
        self.owner_paraphrase_binding_preservation_delta = (
            result.owner_paraphrase_binding_preservation_delta
        )
        self.collapsed_profile_binding_attempted = result.attempted
        self.collapsed_profile_binding_accepted = result.accepted
        self.collapsed_profile_binding_outcome = result.outcome
        self.collapsed_profile_binding_rejection_reason = result.rejection_reason
        self.collapsed_profile_binding_learning_rate_scale = result.learning_rate_scale
        self.collapsed_profile_binding_records = result.records
        self.collapsed_profile_binding_target_profiles = result.target_profiles or []
        self.collapsed_profile_binding_base_score = result.base_score
        self.collapsed_profile_binding_score = result.score
        self.collapsed_profile_binding_delta = result.delta

    def apply_missing_first_token(self, result: Any) -> None:
        self.floor_preserved = result.floor_preserved
        self.profile_probe_snapshot = result.profile_probe_snapshot
        self.profile_score = result.profile_score
        self.diversity_outcome = result.diversity_outcome
        self.diversity_rejection_reason = result.diversity_rejection_reason
        self.coverage_delta = result.coverage_delta
        self.coverage_outcome = result.coverage_outcome
        self.coverage_rejection_reason = result.coverage_rejection_reason
        self.missing_first_token_attempted = result.attempted
        self.missing_first_token_accepted = result.accepted
        self.missing_first_token_outcome = result.outcome
        self.missing_first_token_rejection_reason = result.rejection_reason
        self.missing_first_token_learning_rate_scale = result.learning_rate_scale
        self.missing_first_token_records = result.records
        self.missing_first_token_target_profiles = result.target_profiles or []
        self.missing_first_token_target_ids = result.target_ids or []
        self.missing_first_token_base_score = result.base_score
        self.missing_first_token_score = result.score
        self.missing_first_token_delta = result.delta

    def apply_owner_paraphrase_preservation(
        self,
        preservation: OwnerParaphraseBindingPreservation,
    ) -> None:
        self.owner_paraphrase_binding_preserved = preservation.preserved
        self.owner_paraphrase_binding_preservation_delta = preservation.delta
        if not preservation.preserved:
            self.diversity_rejection_reason = preservation.rejection_reason

    def diversity_accepted(self, diversity_active: bool) -> bool:
        return not diversity_active or self.diversity_outcome in {"improved", "tied"}

    def enforce_coverage_tie_requirement(self, coverage_prep_active: bool) -> None:
        if (
            coverage_prep_active
            and self.floor_preserved
            and self.coverage_outcome == "tied"
            and not self.coverage_prep_accepted
        ):
            self.coverage_rejection_reason = "coverage_tie_without_score_gain"

    def coverage_accepted(self, coverage_frontier_active: bool) -> bool:
        return (
            not coverage_frontier_active
            or self.coverage_outcome == "gained"
            or self.coverage_prep_accepted
        )

    def accepted(self, diversity_accepted: bool, coverage_accepted: bool) -> bool:
        return self.floor_preserved and diversity_accepted and coverage_accepted
