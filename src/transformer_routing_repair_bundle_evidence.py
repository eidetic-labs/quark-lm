"""Evidence checks for declared routing-repair experiment bundles."""

from __future__ import annotations

from typing import Any

from branch_diversity_snapshot_coverage import (
    branch_diversity_snapshot_preserves_target_coverage,
    branch_diversity_snapshot_target_coverage_delta,
)
from constraint_first_report import promotion_check
from transformer_routing_repair_rank_bundle_evidence import rank_routing_repair_checks
from transformer_routing_repair_rank_bundle_evidence import (
    rank_collapse_routing_repair_checks,
)
from transformer_routing_repair_retention_bundle_evidence import (
    retention_rank_routing_repair_checks,
)
from transformer_routing_repair_topk_bundle_evidence import topk_routing_repair_checks
from transformer_routing_repair_bundle import (
    PROFILE_BALANCED_RANK_ROUTING_REPAIR_BUNDLE,
    PROFILE_BALANCED_RANK_COLLAPSE_ROUTING_REPAIR_BUNDLE,
    PROFILE_BALANCED_RETENTION_RANK_ROUTING_REPAIR_BUNDLE,
    PROFILE_BALANCED_TOPK_ROUTING_REPAIR_BUNDLE,
    PROFILE_BALANCED_ROUTING_REPAIR_BUNDLE,
)


def routing_repair_bundle_checks(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Return promotion-shaped evidence checks for declared routing repair bundles."""

    bundle = metrics.get("experiment_bundle")
    if bundle not in {
        PROFILE_BALANCED_ROUTING_REPAIR_BUNDLE,
        PROFILE_BALANCED_RANK_ROUTING_REPAIR_BUNDLE,
        PROFILE_BALANCED_RANK_COLLAPSE_ROUTING_REPAIR_BUNDLE,
        PROFILE_BALANCED_RETENTION_RANK_ROUTING_REPAIR_BUNDLE,
        PROFILE_BALANCED_TOPK_ROUTING_REPAIR_BUNDLE,
    }:
        return []
    if bundle == PROFILE_BALANCED_RANK_ROUTING_REPAIR_BUNDLE:
        return rank_routing_repair_checks(metrics)
    if bundle == PROFILE_BALANCED_RANK_COLLAPSE_ROUTING_REPAIR_BUNDLE:
        return rank_collapse_routing_repair_checks(metrics)
    if bundle == PROFILE_BALANCED_RETENTION_RANK_ROUTING_REPAIR_BUNDLE:
        return retention_rank_routing_repair_checks(metrics)
    if bundle == PROFILE_BALANCED_TOPK_ROUTING_REPAIR_BUNDLE:
        return topk_routing_repair_checks(metrics)
    return _hidden_projection_routing_repair_checks(metrics)


def _hidden_projection_routing_repair_checks(
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    direct_answer = _as_dict(metrics.get("direct_answer"))
    baseline, final = _direct_answer_snapshots(metrics)
    diversity = _as_dict(final.get("branch_diversity_target"))
    audit = _as_dict(final.get("branch_routing_audit"))
    representation = _as_dict(audit.get("representation"))
    logit_prior = _as_dict(audit.get("logit_prior"))
    batch_evidence = _as_dict(direct_answer.get("routing_repair_batch_evidence"))
    coverage_delta = branch_diversity_snapshot_target_coverage_delta(final, baseline)

    hidden_pressure_count = _hidden_projection_pressure_count(logit_prior)
    coverage_surface_present = _coverage_surface_present(coverage_delta)
    diversity_passed = diversity.get("passed") is True
    coverage_preserved = (
        coverage_surface_present
        and branch_diversity_snapshot_preserves_target_coverage(final, baseline)
    )
    coverage_responded = (
        _as_int(coverage_delta.get("improved_profile_count")) > 0 or diversity_passed
    )

    return [
        promotion_check(
            "profile_balanced_branch_batches",
            _profile_balanced_branch_batches_recorded(batch_evidence),
            (
                "Bundle A must record profile-balanced training-family branch "
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
            "hidden_projection_margin_pressure",
            hidden_pressure_count > 0,
            (
                "Bundle A must record hidden-projection or mixed hidden/bias "
                "pressure for branch-routing failures."
            ),
            {
                "hidden_projection_pressure_count": hidden_pressure_count,
                "profile_count": logit_prior.get("profile_count", 0),
                "pressure_counts": logit_prior.get("pressure_counts", {}),
            },
        ),
        promotion_check(
            "representation_separation_evidence",
            _representation_separation_recorded(final, representation),
            (
                "Bundle A must record representation-separation evidence for "
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
                "Bundle A must reject updates that drop final target-token "
                "coverage below baseline coverage."
            ),
            coverage_delta,
        ),
        promotion_check(
            "branch_diversity_acceptance_gate",
            diversity_passed,
            "Bundle A accepts only when branch-diversity evidence passes.",
            {"branch_diversity_target": diversity},
        ),
        promotion_check(
            "hidden_advantage_requires_coverage_response",
            hidden_pressure_count > 0 and coverage_responded,
            (
                "Bundle A must reject hidden-advantage movement when target-token "
                "coverage remains unchanged."
            ),
            {
                "hidden_projection_pressure_count": hidden_pressure_count,
                "coverage_delta": coverage_delta,
                "branch_diversity_passed": diversity_passed,
            },
        ),
    ]


def _direct_answer_snapshots(
    metrics: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    direct_answer = _as_dict(metrics.get("direct_answer"))
    return _as_dict(direct_answer.get("baseline")), _as_dict(direct_answer.get("final"))


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


def _hidden_projection_pressure_count(logit_prior: dict[str, Any]) -> int:
    return _as_int(logit_prior.get("hidden_projection_profile_count")) + _as_int(
        logit_prior.get("mixed_profile_count")
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
