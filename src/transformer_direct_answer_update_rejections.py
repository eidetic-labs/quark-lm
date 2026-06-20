"""Rejected direct-answer update diagnostics."""

from __future__ import annotations

from typing import Any

from branch_diversity_snapshot_coverage import (
    branch_diversity_snapshot_target_coverage_by_profile,
    branch_diversity_snapshot_target_coverage_diagnostics,
)
from branch_diversity_snapshot_stability import (
    branch_diversity_snapshot_stability_diagnostics,
)


def record_direct_update_guard_rejection_attempt(
    direct_answer_update_guard: dict[str, Any],
    direct_baseline: dict[str, Any],
    direct_step: int,
    probe_snapshot: dict[str, Any],
    learning_rate_scale: float,
    update_shape: str = "direct",
) -> None:
    direct_answer_update_guard["rejected_attempts"] += 1
    _record_rejected_scale_and_shape(
        direct_answer_update_guard,
        learning_rate_scale,
        update_shape,
    )
    floor_diagnostics = branch_diversity_snapshot_target_coverage_diagnostics(
        probe_snapshot,
        direct_baseline,
    )
    stability_diagnostics = branch_diversity_snapshot_stability_diagnostics(
        probe_snapshot,
        direct_baseline,
    )
    _record_floor_profile_counts(direct_answer_update_guard, floor_diagnostics)
    _record_stability_counts(direct_answer_update_guard, stability_diagnostics)
    _record_worst_floor_violation(direct_answer_update_guard, floor_diagnostics)
    _record_floor_sample(
        direct_answer_update_guard,
        floor_diagnostics,
        direct_step,
        learning_rate_scale,
        update_shape,
    )
    _record_stability_sample(
        direct_answer_update_guard,
        stability_diagnostics,
        direct_step,
        learning_rate_scale,
        update_shape,
    )
    _record_rejected_step_sample(
        direct_answer_update_guard,
        floor_diagnostics,
        stability_diagnostics,
        direct_step,
        learning_rate_scale,
        update_shape,
        probe_snapshot,
    )


def _record_rejected_scale_and_shape(
    direct_answer_update_guard: dict[str, Any],
    learning_rate_scale: float,
    update_shape: str,
) -> None:
    scale_key = f"{learning_rate_scale:g}"
    rejected_scale_counts = direct_answer_update_guard[
        "rejected_learning_rate_scale_counts"
    ]
    if isinstance(rejected_scale_counts, dict):
        rejected_scale_counts[scale_key] = (
            int(rejected_scale_counts.get(scale_key, 0)) + 1
        )
    rejected_shape_counts = direct_answer_update_guard["rejected_update_shape_counts"]
    if isinstance(rejected_shape_counts, dict):
        rejected_shape_counts[update_shape] = (
            int(rejected_shape_counts.get(update_shape, 0)) + 1
        )


def _record_floor_profile_counts(
    direct_answer_update_guard: dict[str, Any],
    floor_diagnostics: dict[str, Any],
) -> None:
    violation_profile_counts = direct_answer_update_guard[
        "rejected_violation_profile_counts"
    ]
    if not isinstance(violation_profile_counts, dict):
        return
    for violation in floor_diagnostics["violations"]:
        profile = str(violation["profile"])
        violation_profile_counts[profile] = (
            int(violation_profile_counts.get(profile, 0)) + 1
        )


def _record_stability_counts(
    direct_answer_update_guard: dict[str, Any],
    stability_diagnostics: dict[str, Any],
) -> None:
    violation_counts = direct_answer_update_guard.setdefault(
        "rejected_stability_violation_counts",
        {},
    )
    if isinstance(violation_counts, dict):
        for violation in stability_diagnostics["violations"]:
            reason = str(violation["reason"])
            violation_counts[reason] = int(violation_counts.get(reason, 0)) + 1
    profile_counts = direct_answer_update_guard.setdefault(
        "rejected_stability_violation_profile_counts",
        {},
    )
    if isinstance(profile_counts, dict):
        for violation in stability_diagnostics["violations"]:
            profile = str(violation["profile"])
            profile_counts[profile] = int(profile_counts.get(profile, 0)) + 1


def _record_worst_floor_violation(
    direct_answer_update_guard: dict[str, Any],
    floor_diagnostics: dict[str, Any],
) -> None:
    worst_deficit = float(floor_diagnostics["worst_deficit"])
    if worst_deficit <= float(
        direct_answer_update_guard["worst_rejected_coverage_deficit"]
    ):
        return
    direct_answer_update_guard["worst_rejected_coverage_deficit"] = worst_deficit
    direct_answer_update_guard["worst_rejected_coverage_violation"] = (
        floor_diagnostics["worst_violation"]
    )


def _record_floor_sample(
    direct_answer_update_guard: dict[str, Any],
    floor_diagnostics: dict[str, Any],
    direct_step: int,
    learning_rate_scale: float,
    update_shape: str,
) -> None:
    diagnostic_sample = direct_answer_update_guard["rejected_floor_diagnostic_sample"]
    if not isinstance(diagnostic_sample, list) or len(diagnostic_sample) >= 12:
        return
    diagnostic_sample.append(
        {
            "step": direct_step,
            "learning_rate_scale": learning_rate_scale,
            "update_shape": update_shape,
            "preserved": floor_diagnostics["preserved"],
            "violating_profile_count": floor_diagnostics["violating_profile_count"],
            "worst_deficit": floor_diagnostics["worst_deficit"],
            "worst_violation": floor_diagnostics["worst_violation"],
            "violations": floor_diagnostics["violations"][:5],
        }
    )


def _record_stability_sample(
    direct_answer_update_guard: dict[str, Any],
    stability_diagnostics: dict[str, Any],
    direct_step: int,
    learning_rate_scale: float,
    update_shape: str,
) -> None:
    stability_sample = direct_answer_update_guard.setdefault(
        "rejected_stability_diagnostic_sample",
        [],
    )
    if not isinstance(stability_sample, list) or len(stability_sample) >= 12:
        return
    stability_sample.append(
        {
            "step": direct_step,
            "learning_rate_scale": learning_rate_scale,
            "update_shape": update_shape,
            "preserved": stability_diagnostics["preserved"],
            "violating_profile_count": stability_diagnostics[
                "violating_profile_count"
            ],
            "newly_collapsed_profile_count": stability_diagnostics[
                "newly_collapsed_profile_count"
            ],
            "predicted_unique_regression_count": stability_diagnostics[
                "predicted_unique_regression_count"
            ],
            "dominant_rate_regression_count": stability_diagnostics[
                "dominant_rate_regression_count"
            ],
            "worst_violation": stability_diagnostics["worst_violation"],
            "violations": stability_diagnostics["violations"][:5],
        }
    )


def _record_rejected_step_sample(
    direct_answer_update_guard: dict[str, Any],
    floor_diagnostics: dict[str, Any],
    stability_diagnostics: dict[str, Any],
    direct_step: int,
    learning_rate_scale: float,
    update_shape: str,
    probe_snapshot: dict[str, Any],
) -> None:
    rejected_sample = direct_answer_update_guard["rejected_step_sample"]
    if not isinstance(rejected_sample, list) or len(rejected_sample) >= 12:
        return
    rejected_sample.append(
        {
            "step": direct_step,
            "learning_rate_scale": learning_rate_scale,
            "update_shape": update_shape,
            "coverage": branch_diversity_snapshot_target_coverage_by_profile(
                probe_snapshot
            ),
            "floor_diagnostics": _floor_summary(floor_diagnostics),
            "stability_diagnostics": _stability_summary(stability_diagnostics),
        }
    )


def _floor_summary(floor_diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {
        "preserved": floor_diagnostics["preserved"],
        "violating_profile_count": floor_diagnostics["violating_profile_count"],
        "worst_deficit": floor_diagnostics["worst_deficit"],
        "worst_violation": floor_diagnostics["worst_violation"],
    }


def _stability_summary(stability_diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {
        "preserved": stability_diagnostics["preserved"],
        "violating_profile_count": stability_diagnostics["violating_profile_count"],
        "newly_collapsed_profile_count": stability_diagnostics[
            "newly_collapsed_profile_count"
        ],
        "predicted_unique_regression_count": stability_diagnostics[
            "predicted_unique_regression_count"
        ],
        "dominant_rate_regression_count": stability_diagnostics[
            "dominant_rate_regression_count"
        ],
        "worst_violation": stability_diagnostics["worst_violation"],
    }
