"""Evidence checks for profile-balanced rank routing repair."""

from __future__ import annotations

from typing import Any

from branch_diversity_snapshot_coverage import (
    branch_diversity_snapshot_preserves_target_coverage,
    branch_diversity_snapshot_target_coverage_delta,
)
from branch_diversity_snapshots import branch_diversity_snapshot_score_improved
from constraint_first_report import promotion_check
from transformer_routing_repair_bundle import (
    PROFILE_BALANCED_RANK_ROUTING_REPAIR_MODE,
)


def rank_routing_repair_checks(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    direct_answer = _as_dict(metrics.get("direct_answer"))
    baseline = _as_dict(direct_answer.get("baseline"))
    final = _as_dict(direct_answer.get("final"))
    diversity = _as_dict(final.get("branch_diversity_target"))
    audit = _as_dict(final.get("branch_routing_audit"))
    representation = _as_dict(audit.get("representation"))
    batch_evidence = _as_dict(direct_answer.get("routing_repair_batch_evidence"))
    coverage_delta = branch_diversity_snapshot_target_coverage_delta(final, baseline)

    diversity_passed = diversity.get("passed") is True
    coverage_surface_present = _coverage_surface_present(coverage_delta)
    coverage_preserved = (
        coverage_surface_present
        and branch_diversity_snapshot_preserves_target_coverage(final, baseline)
    )
    branch_responded = _branch_response_recorded(
        final,
        baseline,
        coverage_delta,
        diversity_passed,
    )

    return [
        promotion_check(
            "profile_balanced_branch_batches",
            _profile_balanced_branch_batches_recorded(batch_evidence),
            (
                "Bundle B must record profile-balanced training-family branch "
                "batches before judging routing-repair evidence."
            ),
            {
                "batch_evidence": batch_evidence,
                "profile_count": len(_as_dict(final.get("branch_profiles"))),
                "multi_target_profiles": diversity.get("multi_target_profiles", 0),
                "failed_profiles": diversity.get("failed_profiles", 0),
            },
        ),
        promotion_check(
            "rank_margin_pressure",
            _rank_margin_pressure_configured(direct_answer),
            (
                "Bundle B must apply profile-balanced hard-negative "
                "rank-margin pressure for branch-routing failures."
            ),
            {
                "direct_answer_mode": direct_answer.get("direct_answer_mode"),
                "direct_answer_contrast_weight": direct_answer.get(
                    "direct_answer_contrast_weight",
                    0.0,
                ),
                "direct_answer_hard_negatives": direct_answer.get(
                    "direct_answer_hard_negatives",
                    0,
                ),
            },
        ),
        promotion_check(
            "representation_separation_evidence",
            _representation_separation_recorded(final, representation),
            (
                "Bundle B must record representation-separation evidence for "
                "multi-target branch profiles."
            ),
            {
                "profile_count": representation.get("profile_count", 0),
                "low_separation_profile_count": representation.get(
                    "low_separation_profile_count",
                    0,
                ),
                "recorded_profile_count": len(
                    _as_dict(final.get("branch_representation_profiles"))
                ),
            },
        ),
        promotion_check(
            "coverage_preserving_update_guard",
            coverage_preserved,
            (
                "Bundle B must reject updates that drop final target-token "
                "coverage below baseline coverage."
            ),
            coverage_delta,
        ),
        promotion_check(
            "branch_diversity_acceptance_gate",
            diversity_passed,
            "Bundle B accepts only when branch-diversity evidence passes.",
            {"branch_diversity_target": diversity},
        ),
        promotion_check(
            "rank_pressure_requires_branch_response",
            _rank_margin_pressure_configured(direct_answer) and branch_responded,
            (
                "Bundle B must reject rank-margin movement when target coverage "
                "and target-rank branch-diversity score remain unchanged."
            ),
            {
                "coverage_delta": coverage_delta,
                "branch_diversity_passed": diversity_passed,
                "branch_diversity_score_improved": (
                    branch_diversity_snapshot_score_improved(final, baseline)
                ),
            },
        ),
    ]


def _profile_balanced_branch_batches_recorded(
    batch_evidence: dict[str, Any],
) -> bool:
    check = _as_dict(batch_evidence.get("profile_balanced_branch_batches"))
    return check.get("passed") is True


def _representation_separation_recorded(
    final: dict[str, Any],
    representation: dict[str, Any],
) -> bool:
    recorded_profiles = _as_dict(final.get("branch_representation_profiles"))
    return bool(recorded_profiles) and _as_int(representation.get("profile_count")) > 0


def _rank_margin_pressure_configured(direct_answer: dict[str, Any]) -> bool:
    return (
        direct_answer.get("direct_answer_mode")
        == PROFILE_BALANCED_RANK_ROUTING_REPAIR_MODE
        and _as_float(direct_answer.get("direct_answer_contrast_weight")) > 0.0
    )


def _branch_response_recorded(
    final: dict[str, Any],
    baseline: dict[str, Any],
    coverage_delta: dict[str, Any],
    diversity_passed: bool,
) -> bool:
    return (
        diversity_passed
        or _as_int(coverage_delta.get("improved_profile_count")) > 0
        or branch_diversity_snapshot_score_improved(final, baseline)
    )


def _coverage_surface_present(coverage_delta: dict[str, Any]) -> bool:
    return (
        _as_int(coverage_delta.get("baseline_profile_count")) > 0
        and _as_int(coverage_delta.get("snapshot_profile_count")) > 0
    )


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
