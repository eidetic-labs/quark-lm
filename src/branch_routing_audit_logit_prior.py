"""Logit-prior pressure audit for branch routing."""

from __future__ import annotations

from collections import Counter
from typing import Any

from branch_diversity_diagnostics import _as_float, _as_int


def branch_logit_prior_audit_summary(
    branch_profiles: dict[str, dict[str, Any]],
    branch_logit_prior_profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    logit_prior_profiles: list[dict[str, Any]] = []
    pressure_counts: Counter[str] = Counter()
    for name, profile in sorted(branch_logit_prior_profiles.items()):
        branch_profile = branch_profiles.get(name, {})
        target_unique = _as_int(branch_profile.get("diversity", {}).get("target_unique"))
        if target_unique < 2:
            continue
        decomposition = profile.get("dominant_vs_target_decomposition", {})
        failed_summary = decomposition.get("failed_records", {})
        summary = (
            failed_summary
            if _as_int(failed_summary.get("count"))
            else decomposition.get("all_records", {})
        )
        if not summary:
            continue
        pressure = str(summary.get("primary_pressure", "unknown"))
        pressure_counts[pressure] += 1
        logit_prior_profiles.append(
            {
                "profile": name,
                "dominant_predicted_token": profile.get("dominant_predicted_token"),
                "dominant_predicted_rate": _as_float(
                    profile.get("dominant_predicted_rate")
                ),
                "dominant_token_bias": _as_float(profile.get("dominant_token_bias")),
                "dominant_token_bias_rank": profile.get("dominant_token_bias_rank"),
                "avg_bias_advantage": _as_float(summary.get("avg_bias_advantage")),
                "avg_hidden_advantage": _as_float(summary.get("avg_hidden_advantage")),
                "avg_logit_advantage": _as_float(summary.get("avg_logit_advantage")),
                "bias_share_of_positive_advantage": _as_float(
                    summary.get("bias_share_of_positive_advantage")
                ),
                "dominant_logit_win_rate": _as_float(
                    summary.get("dominant_logit_win_rate")
                ),
                "primary_pressure": pressure,
            }
        )

    return {
        "profile_count": len(logit_prior_profiles),
        "pressure_counts": dict(sorted(pressure_counts.items())),
        "bias_driven_profile_count": int(pressure_counts.get("output_bias", 0)),
        "hidden_projection_profile_count": int(
            pressure_counts.get("hidden_projection", 0)
        ),
        "mixed_profile_count": int(pressure_counts.get("mixed_bias_hidden", 0)),
        "profiles": logit_prior_profiles,
    }

