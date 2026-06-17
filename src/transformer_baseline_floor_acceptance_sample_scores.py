"""Score and coverage fields for accepted baseline-floor probe samples."""

from __future__ import annotations

from typing import Any

from transformer_baseline_floor_acceptance_sample_types import (
    BaselineFloorProfileAcceptanceSample,
)


def add_score_and_coverage_fields(
    sample: dict[str, Any],
    sample_input: BaselineFloorProfileAcceptanceSample,
) -> None:
    if (
        sample_input.diversity_active
        and sample_input.profile_score is not None
        and sample_input.profile_base_score is not None
    ):
        sample["diversity_outcome"] = sample_input.diversity_outcome
        sample["base_score"] = list(sample_input.profile_base_score)
        sample["candidate_score"] = list(sample_input.profile_score)
    if sample_input.coverage_active and sample_input.coverage_delta is not None:
        sample["coverage_outcome"] = sample_input.coverage_outcome
        sample["coverage_prep_accepted"] = sample_input.coverage_prep_accepted
        sample["coverage_delta"] = sample_input.coverage_delta
