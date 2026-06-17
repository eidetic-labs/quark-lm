"""Acceptance accounting payloads for baseline-floor profile-scale updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BaselineFloorProfileAcceptanceAccounting:
    profile: str
    scale_key: str
    remaining_profile_binding_prioritized: bool = False
    owner_paraphrase_binding_prioritized: bool = False
    memory_consolidation_prioritized: bool = False
    diversity_active: bool = False
    diversity_outcome: str = "not_active"
    frontier_active: bool = False
    coverage_frontier_active: bool = False
    coverage_outcome: str = "not_active"
    coverage_delta: dict[str, Any] | None = None
    coverage_prep_active: bool = False
    coverage_prep_accepted: bool = False
    coverage_recovery_active: bool = False
    coverage_recovery_attempted: bool = False
    coverage_recovery_accepted: bool = False
    branch_stable_coverage_recovery_active: bool = False
    coverage_recovery_branch_stable_accepted: bool = False
    branch_diversity_recovery_active: bool = False
    branch_diversity_recovery_attempted: bool = False
    branch_diversity_recovery_accepted: bool = False
    branch_diversity_recovery_base_score: tuple[float, ...] | None = None
    branch_diversity_recovery_score: tuple[float, ...] | None = None
    branch_diversity_recovery_outcome: str = "not_active"
    collapsed_profile_binding_active: bool = False
    collapsed_profile_binding_attempted: bool = False
    collapsed_profile_binding_accepted: bool = False
    collapsed_profile_binding_target_profiles: Any = None
    collapsed_profile_binding_base_score: tuple[float, ...] | None = None
    collapsed_profile_binding_score: tuple[float, ...] | None = None
    collapsed_profile_binding_delta: dict[str, Any] | None = None
    collapsed_profile_binding_outcome: str = "not_active"
    missing_first_token_active: bool = False
    missing_first_token_attempted: bool = False
    missing_first_token_accepted: bool = False
    missing_first_token_target_profiles: Any = None
    missing_first_token_target_ids: Any = None
    missing_first_token_base_score: tuple[float, ...] | None = None
    missing_first_token_score: tuple[float, ...] | None = None
    missing_first_token_delta: dict[str, Any] | None = None
    missing_first_token_outcome: str = "not_active"
