"""Branch-routing audit assembly for branch-diversity failures."""

from __future__ import annotations

from collections import Counter
from typing import Any

from branch_diversity_diagnostics import (
    _as_int,
    branch_diversity_root_cause_summary,
)
from branch_routing_audit_logit_prior import branch_logit_prior_audit_summary
from branch_routing_audit_representation import branch_representation_audit_summary


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
    branch_logit_prior_profiles: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Audit likely routing causes behind branch-diversity failure."""

    root_cause = branch_diversity_root_cause_summary(branch_profiles)
    branch_representation_profiles = branch_representation_profiles or {}
    output_bias_by_token = output_bias_by_token or {}
    branch_logit_prior_profiles = branch_logit_prior_profiles or {}

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

    representation = branch_representation_audit_summary(
        branch_representation_profiles
    )
    logit_prior = branch_logit_prior_audit_summary(
        branch_profiles,
        branch_logit_prior_profiles,
    )
    low_separation_profiles = representation["low_separation_profiles"]
    pressure_counts = logit_prior["pressure_counts"]

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
    if pressure_counts.get("output_bias") or pressure_counts.get("mixed_bias_hidden"):
        next_checks.append(
            "Trace output-bias update paths for reused dominant tokens before another repair objective."
        )
    if pressure_counts.get("hidden_projection") or pressure_counts.get("mixed_bias_hidden"):
        next_checks.append(
            "Trace hidden-projection contributions for missing target tokens before another repair objective."
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
        "representation": representation,
        "logit_prior": logit_prior,
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
