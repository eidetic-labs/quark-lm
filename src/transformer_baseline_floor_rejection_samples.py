"""Rejected profile-scale baseline-floor probe samples."""

from __future__ import annotations

from typing import Any

from transformer_baseline_floor_probe_samples import (
    append_baseline_floor_probe_sample,
)


def record_baseline_floor_profile_rejection_sample(
    update_guard: dict[str, Any],
    *,
    profile: str,
    records: int,
    frontier_records: int,
    learning_rate_scale: float,
    scale_key: str,
    diagnostics: dict[str, Any],
    diversity_active: bool,
    profile_score: tuple[float, ...] | None,
    profile_base_score: tuple[float, ...] | None,
    diversity_outcome: str,
    diversity_rejection_reason: str,
    coverage_active: bool,
    coverage_delta: dict[str, Any] | None,
    coverage_outcome: str,
    coverage_prep_accepted: bool,
    coverage_rejection_reason: str,
) -> None:
    rejected_counts = update_guard["sequential_profile_rejection_counts"]
    if isinstance(rejected_counts, dict):
        rejected_counts[profile] = int(rejected_counts.get(profile, 0)) + 1
    scale_counts = update_guard["profile_scale_rejection_scale_counts"]
    if isinstance(scale_counts, dict):
        scale_counts[scale_key] = int(scale_counts.get(scale_key, 0)) + 1
    sample: dict[str, Any] = {
        "profile": profile,
        "accepted": False,
        "records": records,
        "frontier_records": frontier_records,
        "learning_rate_scale": learning_rate_scale,
        "worst_violation": diagnostics["worst_violation"],
        "violating_profile_count": diagnostics["violating_profile_count"],
    }
    if (
        diversity_active
        and profile_score is not None
        and profile_base_score is not None
    ):
        sample["diversity_outcome"] = diversity_outcome
        sample["diversity_rejection_reason"] = diversity_rejection_reason
        sample["base_score"] = list(profile_base_score)
        sample["candidate_score"] = list(profile_score)
    if coverage_active and coverage_delta is not None:
        sample["coverage_outcome"] = coverage_outcome
        sample["coverage_prep_accepted"] = coverage_prep_accepted
        sample["coverage_rejection_reason"] = coverage_rejection_reason
        sample["coverage_delta"] = coverage_delta
    append_baseline_floor_probe_sample(update_guard, sample)
