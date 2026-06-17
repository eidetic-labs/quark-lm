"""Missing-first-token fields for accepted baseline-floor probe samples."""

from __future__ import annotations

from typing import Any

from transformer_baseline_floor_acceptance_sample_types import (
    BaselineFloorProfileAcceptanceSample,
)


def add_missing_first_token_fields(
    sample: dict[str, Any],
    sample_input: BaselineFloorProfileAcceptanceSample,
) -> None:
    if not sample_input.missing_first_token_active:
        return
    sample["missing_first_token_attempted"] = sample_input.missing_first_token_attempted
    sample["missing_first_token_accepted"] = sample_input.missing_first_token_accepted
    sample["missing_first_token_outcome"] = sample_input.missing_first_token_outcome
    sample["missing_first_token_target_profiles"] = (
        sample_input.missing_first_token_target_profiles
    )
    sample["missing_first_token_target_ids"] = (
        sample_input.missing_first_token_target_ids
    )
    sample["missing_first_token_profile_specific"] = (
        sample_input.missing_first_token_profile_specific
    )
    if sample_input.missing_first_token_rejection_reason:
        sample["missing_first_token_rejection_reason"] = (
            sample_input.missing_first_token_rejection_reason
        )
    if sample_input.missing_first_token_learning_rate_scale is not None:
        sample["missing_first_token_learning_rate_scale"] = (
            sample_input.missing_first_token_learning_rate_scale
        )
    sample["missing_first_token_records"] = sample_input.missing_first_token_records
    if sample_input.missing_first_token_base_score is not None:
        sample["missing_first_token_base_score"] = list(
            sample_input.missing_first_token_base_score
        )
    if sample_input.missing_first_token_score is not None:
        sample["missing_first_token_score"] = list(
            sample_input.missing_first_token_score
        )
    if sample_input.missing_first_token_delta is not None:
        sample["missing_first_token_delta"] = sample_input.missing_first_token_delta
