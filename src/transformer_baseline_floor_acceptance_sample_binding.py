"""Binding fields for accepted baseline-floor probe samples."""

from __future__ import annotations

from typing import Any

from transformer_baseline_floor_acceptance_sample_types import (
    BaselineFloorProfileAcceptanceSample,
)


def add_binding_fields(
    sample: dict[str, Any],
    sample_input: BaselineFloorProfileAcceptanceSample,
) -> None:
    if sample_input.remaining_profile_binding_active:
        sample["remaining_profile_binding_prioritized"] = (
            sample_input.remaining_profile_binding_prioritized
        )
        sample["remaining_profile_binding_target_profiles"] = (
            sample_input.remaining_profile_binding_target_profiles
        )
        sample["remaining_profile_binding_source_profiles"] = (
            sample_input.remaining_profile_binding_source_profiles
        )
    if sample_input.owner_paraphrase_binding_active:
        sample["owner_paraphrase_binding_prioritized"] = (
            sample_input.owner_paraphrase_binding_prioritized
        )
        sample["owner_paraphrase_binding_target_profiles"] = (
            sample_input.owner_paraphrase_binding_target_profiles
        )
        sample["owner_paraphrase_binding_preserved_profiles"] = (
            sample_input.owner_paraphrase_binding_preserved_profiles
        )
        sample["owner_paraphrase_binding_preserved"] = (
            sample_input.owner_paraphrase_binding_preserved
        )
        if sample_input.owner_paraphrase_binding_preservation_delta is not None:
            sample["owner_paraphrase_binding_preservation_delta"] = (
                sample_input.owner_paraphrase_binding_preservation_delta
            )
    if sample_input.memory_consolidation_active:
        add_memory_consolidation_fields(sample, sample_input)


def add_memory_consolidation_fields(
    sample: dict[str, Any],
    sample_input: BaselineFloorProfileAcceptanceSample,
) -> None:
    sample["memory_consolidation_prioritized"] = (
        sample_input.memory_consolidation_prioritized
    )
    sample["memory_consolidation_target_profiles"] = (
        sample_input.memory_consolidation_target_profiles
    )
    sample["memory_consolidation_source_plan"] = (
        sample_input.memory_consolidation_source_plan
    )
    sample["memory_consolidation_collapsed_memory_backed_profiles"] = (
        sample_input.memory_consolidation_collapsed_memory_backed_profiles
    )
    if sample_input.memory_consolidation_remaining_collapsed_active:
        sample["memory_consolidation_remaining_collapsed_target_profiles"] = (
            sample_input.memory_consolidation_target_profiles
        )
    if sample_input.memory_consolidation_profile_specific_active:
        sample["memory_consolidation_profile_specific_missing_first_token_target_map"] = (
            sample_input.memory_consolidation_profile_specific_missing_first_token_target_map
        )
