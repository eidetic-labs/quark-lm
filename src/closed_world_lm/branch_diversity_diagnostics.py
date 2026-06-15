"""Branch-diversity root-cause diagnostics.

These helpers are intentionally data-only. They do not change training behavior;
they explain which kind of branch-diversity failure a snapshot is showing.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def branch_diversity_profile_diagnosis(
    name: str,
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Summarize the failure modes for one branch-diversity profile."""

    diversity = profile.get("diversity", {})
    target_rank = profile.get("target_rank", {})
    target_unique = _as_int(diversity.get("target_unique"))
    predicted_unique = _as_int(diversity.get("predicted_unique"))
    target_token_coverage = _as_float(diversity.get("target_token_coverage"))
    dominant_rate = _as_float(diversity.get("dominant_predicted_rate"))
    target_top3_rate = _as_float(target_rank.get("top3_rate"))
    target_top5_rate = _as_float(target_rank.get("top5_rate"))
    avg_target_rank = _as_float(target_rank.get("avg"))
    missing_targets = diversity.get("missing_target_tokens", [])
    missing_target_count = sum(
        _as_int(item.get("count"), 1)
        for item in missing_targets
        if isinstance(item, dict)
    )

    failure_modes: list[str] = []
    if target_unique < 2:
        failure_modes.append("single_target_profile")
    else:
        if predicted_unique <= 1:
            failure_modes.append("prediction_collapse")
        elif predicted_unique < target_unique:
            failure_modes.append("under_diverse_predictions")
        if target_token_coverage <= 0.0:
            failure_modes.append("zero_target_coverage")
        elif target_token_coverage < 1.0:
            failure_modes.append("target_coverage_gap")
        if dominant_rate >= 0.999:
            failure_modes.append("single_dominant_prediction")
        elif dominant_rate >= 0.75:
            failure_modes.append("dominant_prediction_bias")
        if predicted_unique >= target_unique and target_token_coverage < 1.0:
            failure_modes.append("wrong_diverse_predictions")
        if target_token_coverage < 1.0 and target_top5_rate <= 0.25:
            failure_modes.append("targets_buried")
        elif target_token_coverage < 1.0 and target_top3_rate <= 0.25:
            failure_modes.append("targets_not_top3")

    return {
        "name": name,
        "target_unique": target_unique,
        "predicted_unique": predicted_unique,
        "target_token_coverage": target_token_coverage,
        "dominant_predicted_token": diversity.get("dominant_predicted_token"),
        "dominant_predicted_rate": dominant_rate,
        "avg_target_rank": avg_target_rank,
        "target_top3_rate": target_top3_rate,
        "target_top5_rate": target_top5_rate,
        "missing_target_count": missing_target_count,
        "failure_modes": failure_modes,
    }


def branch_diversity_root_cause_summary(
    branch_profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate branch-diversity failure modes across multi-target profiles."""

    profile_diagnoses = [
        branch_diversity_profile_diagnosis(name, profile)
        for name, profile in sorted(branch_profiles.items())
    ]
    multi_target_profiles = [
        diagnosis
        for diagnosis in profile_diagnoses
        if diagnosis["target_unique"] >= 2
    ]
    mode_counts: Counter[str] = Counter()
    dominant_token_profiles: dict[str, list[str]] = defaultdict(list)
    for diagnosis in multi_target_profiles:
        mode_counts.update(diagnosis["failure_modes"])
        dominant_token = diagnosis.get("dominant_predicted_token")
        if dominant_token is not None:
            dominant_token_profiles[str(dominant_token)].append(diagnosis["name"])

    dominant_tokens = [
        {
            "token": token,
            "profile_count": len(profiles),
            "profiles": sorted(profiles),
        }
        for token, profiles in dominant_token_profiles.items()
    ]
    dominant_tokens.sort(
        key=lambda item: (-int(item["profile_count"]), str(item["token"]))
    )
    reused_dominant_tokens = [
        item for item in dominant_tokens if int(item["profile_count"]) > 1
    ]
    profile_count = len(multi_target_profiles)
    collapsed_count = int(mode_counts.get("prediction_collapse", 0))
    zero_coverage_count = int(mode_counts.get("zero_target_coverage", 0))
    buried_count = int(mode_counts.get("targets_buried", 0))
    wrong_diverse_count = int(mode_counts.get("wrong_diverse_predictions", 0))

    if profile_count == 0:
        hypothesis = "no_multi_target_profiles"
        severity = "none"
    elif collapsed_count == profile_count and reused_dominant_tokens:
        hypothesis = "global_output_prior_collapse"
        severity = "critical"
    elif collapsed_count == profile_count:
        hypothesis = "profile_local_prediction_collapse"
        severity = "critical"
    elif zero_coverage_count:
        hypothesis = "target_routing_gap"
        severity = "critical"
    elif buried_count:
        hypothesis = "target_rank_burial"
        severity = "high"
    elif wrong_diverse_count:
        hypothesis = "wrong_diversity_not_target_coverage"
        severity = "high"
    elif mode_counts:
        hypothesis = "mixed_branch_diversity_gap"
        severity = "medium"
    else:
        hypothesis = "no_branch_diversity_gap"
        severity = "none"

    recommendations: list[str] = []
    if reused_dominant_tokens:
        recommendations.append(
            "Audit global logit priors and output-bias escape paths before adding another objective."
        )
    if collapsed_count:
        recommendations.append(
            "Measure prompt-to-branch representation separation before treating the failure as missing-token coverage only."
        )
    if zero_coverage_count:
        recommendations.append(
            "Prioritize target-routing and target-rank lift for zero-coverage profiles before diversity-only pressure."
        )
    if buried_count:
        recommendations.append(
            "Lift correct targets into the top-k set before relying on decoding or target-set diversity."
        )
    if wrong_diverse_count:
        recommendations.append(
            "Require target-aligned coverage, not just more distinct predicted tokens."
        )
    if profile_count:
        recommendations.append(
            "Keep retrieval-memory success separate from weight-consolidation promotion."
        )

    profiles = sorted(
        multi_target_profiles,
        key=lambda item: (
            float(item["target_token_coverage"]),
            -float(item["dominant_predicted_rate"]),
            str(item["name"]),
        ),
    )

    return {
        "profile_count": profile_count,
        "severity": severity,
        "root_cause_hypothesis": hypothesis,
        "mode_counts": dict(sorted(mode_counts.items())),
        "dominant_tokens": dominant_tokens,
        "reused_dominant_tokens": reused_dominant_tokens,
        "global_dominant_token_reuse": bool(reused_dominant_tokens),
        "zero_coverage_profile_count": zero_coverage_count,
        "collapsed_profile_count": collapsed_count,
        "buried_target_profile_count": buried_count,
        "wrong_diverse_profile_count": wrong_diverse_count,
        "recommendations": recommendations,
        "profiles": profiles,
    }
