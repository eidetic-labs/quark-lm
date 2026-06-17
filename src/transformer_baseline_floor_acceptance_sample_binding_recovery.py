"""Collapsed-profile binding fields for accepted baseline-floor probe samples."""

from __future__ import annotations

from typing import Any

from transformer_baseline_floor_acceptance_sample_types import (
    BaselineFloorProfileAcceptanceSample,
)


def add_binding_recovery_fields(
    sample: dict[str, Any],
    sample_input: BaselineFloorProfileAcceptanceSample,
) -> None:
    if not sample_input.collapsed_profile_binding_active:
        return
    sample["collapsed_profile_binding_attempted"] = (
        sample_input.collapsed_profile_binding_attempted
    )
    sample["collapsed_profile_binding_accepted"] = (
        sample_input.collapsed_profile_binding_accepted
    )
    sample["collapsed_profile_binding_outcome"] = (
        sample_input.collapsed_profile_binding_outcome
    )
    sample["collapsed_profile_binding_target_profiles"] = (
        sample_input.collapsed_profile_binding_target_profiles
    )
    if sample_input.collapsed_profile_binding_rejection_reason:
        sample["collapsed_profile_binding_rejection_reason"] = (
            sample_input.collapsed_profile_binding_rejection_reason
        )
    if sample_input.collapsed_profile_binding_learning_rate_scale is not None:
        sample["collapsed_profile_binding_learning_rate_scale"] = (
            sample_input.collapsed_profile_binding_learning_rate_scale
        )
    sample["collapsed_profile_binding_records"] = (
        sample_input.collapsed_profile_binding_records
    )
    if sample_input.collapsed_profile_binding_base_score is not None:
        sample["collapsed_profile_binding_base_score"] = list(
            sample_input.collapsed_profile_binding_base_score
        )
    if sample_input.collapsed_profile_binding_score is not None:
        sample["collapsed_profile_binding_score"] = list(
            sample_input.collapsed_profile_binding_score
        )
    if sample_input.collapsed_profile_binding_delta is not None:
        sample["collapsed_profile_binding_delta"] = (
            sample_input.collapsed_profile_binding_delta
        )
