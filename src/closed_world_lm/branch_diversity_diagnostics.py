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


def _token_counts(items: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for item in items:
        if not isinstance(item, dict):
            continue
        value = item.get("value")
        if value is None:
            continue
        counts[str(value)] += _as_int(item.get("count"), 1)
    return counts


def _bias_rankings(output_bias_by_token: dict[str, float]) -> dict[str, dict[str, Any]]:
    ordered = sorted(
        output_bias_by_token.items(),
        key=lambda item: (-float(item[1]), item[0]),
    )
    return {
        token: {"rank": index + 1, "bias": float(value)}
        for index, (token, value) in enumerate(ordered)
    }


def branch_routing_audit_summary(
    branch_profiles: dict[str, dict[str, Any]],
    branch_representation_profiles: dict[str, dict[str, Any]] | None = None,
    output_bias_by_token: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Audit likely routing causes behind branch-diversity failure."""

    root_cause = branch_diversity_root_cause_summary(branch_profiles)
    branch_representation_profiles = branch_representation_profiles or {}
    output_bias_by_token = output_bias_by_token or {}

    target_counts: Counter[str] = Counter()
    predicted_counts: Counter[str] = Counter()
    profile_imbalance: list[dict[str, Any]] = []
    for name, profile in sorted(branch_profiles.items()):
        diversity = profile.get("diversity", {})
        target_unique = _as_int(diversity.get("target_unique"))
        if target_unique < 2:
            continue
        profile_target_counts = _token_counts(profile.get("target_tokens", []))
        profile_predicted_counts = _token_counts(profile.get("predicted_tokens", []))
        target_counts.update(profile_target_counts)
        predicted_counts.update(profile_predicted_counts)
        total = sum(profile_target_counts.values())
        top_token = None
        top_count = 0
        if profile_target_counts:
            top_token, top_count = profile_target_counts.most_common(1)[0]
        profile_imbalance.append(
            {
                "profile": name,
                "target_unique": target_unique,
                "target_count": total,
                "top_target_token": top_token,
                "top_target_count": top_count,
                "top_target_share": top_count / total if total else 0.0,
            }
        )

    bias_rankings = _bias_rankings(output_bias_by_token)
    dominant_bias_records: list[dict[str, Any]] = []
    for item in root_cause.get("dominant_tokens", []):
        token = str(item.get("token"))
        bias_info = bias_rankings.get(token, {"rank": None, "bias": 0.0})
        dominant_bias_records.append(
            {
                "token": token,
                "profile_count": _as_int(item.get("profile_count")),
                "profiles": item.get("profiles", []),
                "bias": float(bias_info.get("bias", 0.0)),
                "bias_rank": bias_info.get("rank"),
            }
        )
    dominant_bias_records.sort(
        key=lambda item: (
            item["bias_rank"] if item["bias_rank"] is not None else 10**9,
            -int(item["profile_count"]),
            str(item["token"]),
        )
    )
    target_bias_values = [
        float(output_bias_by_token[token])
        for token in target_counts
        if token in output_bias_by_token
    ]
    dominant_bias_values = [
        float(item["bias"]) for item in dominant_bias_records
    ]
    avg_target_bias = (
        sum(target_bias_values) / len(target_bias_values)
        if target_bias_values
        else 0.0
    )
    max_dominant_bias = max(dominant_bias_values) if dominant_bias_values else 0.0
    top_bias_tokens = [
        {"token": token, "bias": float(value), "rank": index + 1}
        for index, (token, value) in enumerate(
            sorted(
                output_bias_by_token.items(),
                key=lambda item: (-float(item[1]), item[0]),
            )[:12]
        )
    ]
    dominant_top3 = [
        item
        for item in dominant_bias_records
        if item["bias_rank"] is not None and int(item["bias_rank"]) <= 3
    ]
    output_bias_escape_risk = (
        "high"
        if dominant_top3 and max_dominant_bias > avg_target_bias + 1e-12
        else "medium"
        if dominant_top3
        else "low"
    )

    representation_profiles: list[dict[str, Any]] = []
    low_separation_profiles: list[dict[str, Any]] = []
    different_distances: list[float] = []
    for name, profile in sorted(branch_representation_profiles.items()):
        target_unique = _as_int(profile.get("target_unique"))
        if target_unique < 2:
            continue
        different_avg = _as_float(
            profile.get("different_target_pairwise_distance", {}).get("avg")
        )
        same_avg = _as_float(
            profile.get("same_target_pairwise_distance", {}).get("avg")
        )
        separation_ratio = (
            different_avg / same_avg
            if same_avg > 1e-12
            else different_avg
        )
        record = {
            "profile": name,
            "target_unique": target_unique,
            "different_target_distance_avg": different_avg,
            "same_target_distance_avg": same_avg,
            "separation_ratio": separation_ratio,
        }
        representation_profiles.append(record)
        different_distances.append(different_avg)
        if different_avg <= 0.01 or separation_ratio <= 1.05:
            low_separation_profiles.append(record)

    target_total = sum(target_counts.values())
    top_target_token = None
    top_target_count = 0
    if target_counts:
        top_target_token, top_target_count = target_counts.most_common(1)[0]
    rare_target_tokens = [
        {"token": token, "count": count}
        for token, count in sorted(target_counts.items())
        if count == 1
    ]
    high_imbalance_profiles = [
        item for item in profile_imbalance if float(item["top_target_share"]) >= 0.5
    ]

    if root_cause.get("root_cause_hypothesis") == "target_routing_gap":
        audit_hypothesis = "routing_gap_requires_representation_and_logit_audit"
    elif output_bias_escape_risk == "high":
        audit_hypothesis = "output_bias_escape_risk"
    elif low_separation_profiles:
        audit_hypothesis = "representation_separation_risk"
    elif high_imbalance_profiles:
        audit_hypothesis = "profile_target_imbalance_risk"
    else:
        audit_hypothesis = "no_single_audit_risk"

    next_checks: list[str] = []
    if output_bias_escape_risk != "low":
        next_checks.append(
            "Compare dominant-token bias ranks against missing target-token bias ranks."
        )
    if low_separation_profiles:
        next_checks.append(
            "Measure whether failed prompts form separable hidden-state clusters by target token."
        )
    if high_imbalance_profiles or rare_target_tokens:
        next_checks.append(
            "Balance candidate construction by profile and target token before another objective screen."
        )
    if root_cause.get("zero_coverage_profile_count", 0):
        next_checks.append(
            "Prioritize zero-coverage profiles before optimizing average diversity."
        )

    return {
        "audit_hypothesis": audit_hypothesis,
        "root_cause_hypothesis": root_cause.get("root_cause_hypothesis"),
        "output_bias": {
            "escape_risk": output_bias_escape_risk,
            "avg_target_bias": avg_target_bias,
            "max_dominant_bias": max_dominant_bias,
            "dominant_bias_advantage": max_dominant_bias - avg_target_bias,
            "dominant_tokens": dominant_bias_records,
            "top_bias_tokens": top_bias_tokens,
        },
        "representation": {
            "profile_count": len(representation_profiles),
            "low_separation_profile_count": len(low_separation_profiles),
            "min_different_target_distance": (
                min(different_distances) if different_distances else 0.0
            ),
            "avg_different_target_distance": (
                sum(different_distances) / len(different_distances)
                if different_distances
                else 0.0
            ),
            "low_separation_profiles": low_separation_profiles,
            "profiles": representation_profiles,
        },
        "target_imbalance": {
            "profile_count": len(profile_imbalance),
            "target_token_count": len(target_counts),
            "total_target_count": target_total,
            "top_target_token": top_target_token,
            "top_target_count": top_target_count,
            "top_target_share": (
                top_target_count / target_total if target_total else 0.0
            ),
            "rare_target_token_count": len(rare_target_tokens),
            "rare_target_tokens": rare_target_tokens[:24],
            "high_imbalance_profiles": high_imbalance_profiles,
            "profiles": profile_imbalance,
        },
        "next_checks": next_checks,
    }
