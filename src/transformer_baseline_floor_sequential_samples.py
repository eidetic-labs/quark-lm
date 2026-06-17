"""Sequential baseline-floor profile probe samples."""

from __future__ import annotations

from typing import Any

from transformer_baseline_floor_probe_samples import (
    append_sample,
    increment_sample_count,
)


def record_baseline_floor_sequential_profile_probe_result(
    update_guard: dict[str, Any],
    *,
    profile: str,
    accepted: bool,
    records: int,
    diagnostics: dict[str, Any] | None = None,
) -> None:
    sample: dict[str, Any] = {
        "profile": profile,
        "accepted": accepted,
        "records": records,
    }
    if accepted:
        update_guard["sequential_profile_acceptances"] += 1
        increment_sample_count(
            update_guard,
            "sequential_profile_acceptance_counts",
            profile,
        )
    else:
        update_guard["sequential_profile_rejections"] += 1
        increment_sample_count(
            update_guard,
            "sequential_profile_rejection_counts",
            profile,
        )
        if diagnostics is not None:
            sample["worst_violation"] = diagnostics["worst_violation"]
            sample["violating_profile_count"] = diagnostics[
                "violating_profile_count"
            ]
    append_sample(update_guard, "sequential_profile_probe_sample", sample)
