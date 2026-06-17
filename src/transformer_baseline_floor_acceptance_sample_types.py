"""Types for accepted baseline-floor profile probe samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformer_baseline_floor_probe_samples import BaselineFloorProbeSampleStreams


@dataclass(frozen=True)
class BaselineFloorProfileAcceptanceSample:
    profile: str
    records: int
    frontier_records: int
    learning_rate_scale: float
    streams: BaselineFloorProbeSampleStreams
    remaining_profile_binding_active: bool = False
    remaining_profile_binding_prioritized: bool = False
    remaining_profile_binding_target_profiles: Any = None
    remaining_profile_binding_source_profiles: Any = None
    owner_paraphrase_binding_active: bool = False
    owner_paraphrase_binding_prioritized: bool = False
    owner_paraphrase_binding_target_profiles: Any = None
    owner_paraphrase_binding_preserved_profiles: Any = None
    owner_paraphrase_binding_preserved: bool = False
    owner_paraphrase_binding_preservation_delta: dict[str, Any] | None = None
    memory_consolidation_active: bool = False
    memory_consolidation_prioritized: bool = False
    memory_consolidation_target_profiles: Any = None
    memory_consolidation_source_plan: str | None = None
    memory_consolidation_collapsed_memory_backed_profiles: Any = None
    memory_consolidation_remaining_collapsed_active: bool = False
    memory_consolidation_profile_specific_active: bool = False
    memory_consolidation_profile_specific_missing_first_token_target_map: Any = None
    diversity_active: bool = False
    diversity_outcome: str = "not_active"
    profile_score: tuple[float, ...] | None = None
    profile_base_score: tuple[float, ...] | None = None
    coverage_active: bool = False
    coverage_outcome: str = "not_active"
    coverage_prep_accepted: bool = False
    coverage_delta: dict[str, Any] | None = None
    coverage_recovery_active: bool = False
    coverage_recovery_attempted: bool = False
    coverage_recovery_accepted: bool = False
    coverage_recovery_outcome: str = "not_active"
    coverage_recovery_records: int = 0
    coverage_recovery_learning_rate_scale: float | None = None
    coverage_recovery_delta: dict[str, Any] | None = None
    branch_stable_coverage_recovery_active: bool = False
    coverage_recovery_branch_stable_checked: bool = False
    coverage_recovery_branch_stable_accepted: bool = False
    coverage_recovery_branch_stability_preserved: bool | None = None
    coverage_recovery_prepared_score: tuple[float, ...] | None = None
    coverage_recovery_score: tuple[float, ...] | None = None
    branch_diversity_recovery_active: bool = False
    branch_diversity_recovery_attempted: bool = False
    branch_diversity_recovery_accepted: bool = False
    branch_diversity_recovery_outcome: str = "not_active"
    branch_diversity_recovery_rejection_reason: str = ""
    branch_diversity_recovery_learning_rate_scale: float | None = None
    branch_diversity_recovery_records: int = 0
    branch_diversity_recovery_base_score: tuple[float, ...] | None = None
    branch_diversity_recovery_score: tuple[float, ...] | None = None
    branch_diversity_recovery_delta: dict[str, Any] | None = None
    collapsed_profile_binding_active: bool = False
    collapsed_profile_binding_attempted: bool = False
    collapsed_profile_binding_accepted: bool = False
    collapsed_profile_binding_outcome: str = "not_active"
    collapsed_profile_binding_target_profiles: Any = None
    collapsed_profile_binding_rejection_reason: str = ""
    collapsed_profile_binding_learning_rate_scale: float | None = None
    collapsed_profile_binding_records: int = 0
    collapsed_profile_binding_base_score: tuple[float, ...] | None = None
    collapsed_profile_binding_score: tuple[float, ...] | None = None
    collapsed_profile_binding_delta: dict[str, Any] | None = None
    missing_first_token_active: bool = False
    missing_first_token_attempted: bool = False
    missing_first_token_accepted: bool = False
    missing_first_token_outcome: str = "not_active"
    missing_first_token_target_profiles: Any = None
    missing_first_token_target_ids: Any = None
    missing_first_token_profile_specific: bool = False
    missing_first_token_rejection_reason: str = ""
    missing_first_token_learning_rate_scale: float | None = None
    missing_first_token_records: int = 0
    missing_first_token_base_score: tuple[float, ...] | None = None
    missing_first_token_score: tuple[float, ...] | None = None
    missing_first_token_delta: dict[str, Any] | None = None
