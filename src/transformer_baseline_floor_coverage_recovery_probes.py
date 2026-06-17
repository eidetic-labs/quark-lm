"""Probe recording for baseline-floor coverage recovery."""

from __future__ import annotations

from typing import Any


def coverage_recovery_probe_metadata(
    *,
    profile_scale: float,
    recovery_learning_rate_scale: float,
    update_shape: str,
    profile: str,
    profile_batch_records: int,
    profile_frontier_records: int,
    recovery_records: int,
) -> dict[str, Any]:
    return {
        "baseline_floor_update_guard_probe": True,
        "baseline_floor_sequential_profile_probe": True,
        "baseline_floor_calibrated_sequential_profile_probe": True,
        "baseline_floor_profile_scale_memory_probe": True,
        "baseline_floor_profile_scale_frontier_probe": True,
        "baseline_floor_profile_scale_coverage_recovery_probe": True,
        "learning_rate_scale": profile_scale,
        "coverage_recovery_learning_rate_scale": recovery_learning_rate_scale,
        "update_shape": update_shape,
        "sequential_profile": profile,
        "sequential_profile_records": profile_batch_records,
        "sequential_profile_frontier_records": profile_frontier_records,
        "coverage_recovery_records": recovery_records,
    }


def record_coverage_recovery_probe(
    *,
    snapshot_recorder: Any,
    direct_step: int,
    profile_scale: float,
    recovery_learning_rate_scale: float,
    update_shape: str,
    profile: str,
    profile_batch_records: int,
    profile_frontier_records: int,
    recovery_records: int,
) -> dict[str, Any]:
    return snapshot_recorder.record(
        direct_step,
        None,
        coverage_recovery_probe_metadata(
            profile_scale=profile_scale,
            recovery_learning_rate_scale=recovery_learning_rate_scale,
            update_shape=update_shape,
            profile=profile,
            profile_batch_records=profile_batch_records,
            profile_frontier_records=profile_frontier_records,
            recovery_records=recovery_records,
        ),
    )
