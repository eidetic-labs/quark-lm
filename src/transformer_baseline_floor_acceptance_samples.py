"""Accepted profile-scale baseline-floor probe samples."""

from __future__ import annotations

from typing import Any

from transformer_baseline_floor_acceptance_sample_binding import add_binding_fields
from transformer_baseline_floor_acceptance_sample_binding_recovery import (
    add_binding_recovery_fields,
)
from transformer_baseline_floor_acceptance_sample_missing_first_token import (
    add_missing_first_token_fields,
)
from transformer_baseline_floor_acceptance_sample_recovery import add_recovery_fields
from transformer_baseline_floor_acceptance_sample_scores import (
    add_score_and_coverage_fields,
)
from transformer_baseline_floor_acceptance_sample_types import (
    BaselineFloorProfileAcceptanceSample,
)
from transformer_baseline_floor_probe_samples import append_baseline_floor_probe_sample


def record_baseline_floor_profile_acceptance_sample(
    update_guard: dict[str, Any],
    sample_input: BaselineFloorProfileAcceptanceSample,
) -> None:
    sample: dict[str, Any] = {
        "profile": sample_input.profile,
        "accepted": True,
        "records": sample_input.records,
        "frontier_records": sample_input.frontier_records,
        "learning_rate_scale": sample_input.learning_rate_scale,
    }
    add_binding_fields(sample, sample_input)
    add_score_and_coverage_fields(sample, sample_input)
    add_recovery_fields(sample, sample_input)
    add_binding_recovery_fields(sample, sample_input)
    add_missing_first_token_fields(sample, sample_input)
    append_baseline_floor_probe_sample(update_guard, sample, sample_input.streams)


__all__ = [
    "BaselineFloorProfileAcceptanceSample",
    "record_baseline_floor_profile_acceptance_sample",
]
