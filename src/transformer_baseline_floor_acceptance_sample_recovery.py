"""Coverage-recovery fields for accepted baseline-floor probe samples."""

from __future__ import annotations

from typing import Any

from transformer_baseline_floor_acceptance_sample_types import (
    BaselineFloorProfileAcceptanceSample,
)


def add_recovery_fields(
    sample: dict[str, Any],
    sample_input: BaselineFloorProfileAcceptanceSample,
) -> None:
    if not (
        sample_input.coverage_recovery_active
        and sample_input.coverage_recovery_attempted
    ):
        return
    sample["coverage_recovery_attempted"] = sample_input.coverage_recovery_attempted
    sample["coverage_recovery_accepted"] = sample_input.coverage_recovery_accepted
    sample["coverage_recovery_outcome"] = sample_input.coverage_recovery_outcome
    sample["coverage_recovery_records"] = sample_input.coverage_recovery_records
    if sample_input.coverage_recovery_learning_rate_scale is not None:
        sample["coverage_recovery_learning_rate_scale"] = (
            sample_input.coverage_recovery_learning_rate_scale
        )
    if sample_input.coverage_recovery_delta is not None:
        sample["coverage_recovery_delta"] = sample_input.coverage_recovery_delta
    add_branch_stable_recovery_fields(sample, sample_input)
    add_branch_diversity_recovery_fields(sample, sample_input)


def add_branch_stable_recovery_fields(
    sample: dict[str, Any],
    sample_input: BaselineFloorProfileAcceptanceSample,
) -> None:
    if not sample_input.branch_stable_coverage_recovery_active:
        return
    sample["coverage_recovery_branch_stable_checked"] = (
        sample_input.coverage_recovery_branch_stable_checked
    )
    sample["coverage_recovery_branch_stable_accepted"] = (
        sample_input.coverage_recovery_branch_stable_accepted
    )
    if sample_input.coverage_recovery_branch_stability_preserved is not None:
        sample["coverage_recovery_branch_stability_preserved"] = (
            sample_input.coverage_recovery_branch_stability_preserved
        )
    if sample_input.coverage_recovery_prepared_score is not None:
        sample["coverage_recovery_prepared_score"] = list(
            sample_input.coverage_recovery_prepared_score
        )
    if sample_input.coverage_recovery_score is not None:
        sample["coverage_recovery_score"] = list(sample_input.coverage_recovery_score)


def add_branch_diversity_recovery_fields(
    sample: dict[str, Any],
    sample_input: BaselineFloorProfileAcceptanceSample,
) -> None:
    if not sample_input.branch_diversity_recovery_active:
        return
    sample["branch_diversity_recovery_attempted"] = (
        sample_input.branch_diversity_recovery_attempted
    )
    sample["branch_diversity_recovery_accepted"] = (
        sample_input.branch_diversity_recovery_accepted
    )
    sample["branch_diversity_recovery_outcome"] = (
        sample_input.branch_diversity_recovery_outcome
    )
    if sample_input.branch_diversity_recovery_rejection_reason:
        sample["branch_diversity_recovery_rejection_reason"] = (
            sample_input.branch_diversity_recovery_rejection_reason
        )
    if sample_input.branch_diversity_recovery_learning_rate_scale is not None:
        sample["branch_diversity_recovery_learning_rate_scale"] = (
            sample_input.branch_diversity_recovery_learning_rate_scale
        )
    sample["branch_diversity_recovery_records"] = (
        sample_input.branch_diversity_recovery_records
    )
    if sample_input.branch_diversity_recovery_base_score is not None:
        sample["branch_diversity_recovery_base_score"] = list(
            sample_input.branch_diversity_recovery_base_score
        )
    if sample_input.branch_diversity_recovery_score is not None:
        sample["branch_diversity_recovery_score"] = list(
            sample_input.branch_diversity_recovery_score
        )
    if sample_input.branch_diversity_recovery_delta is not None:
        sample["branch_diversity_recovery_delta"] = (
            sample_input.branch_diversity_recovery_delta
        )
