"""Branch-diversity target summaries for direct-answer diagnostics."""

from __future__ import annotations

from typing import Any

from branch_diversity_diagnostics import branch_diversity_root_cause_summary


def summarize_branch_diversity_target(
    branch_profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    blocking_evals: list[dict[str, Any]] = []
    multi_target_profiles = 0
    passed_profiles = 0
    max_dominant_rate = 0.0
    min_target_token_coverage = 1.0

    for name, profile in sorted(branch_profiles.items()):
        diversity = profile.get("diversity", {})
        target_unique = int(diversity.get("target_unique", 0))
        predicted_unique = int(diversity.get("predicted_unique", 0))
        target_token_coverage = float(diversity.get("target_token_coverage", 0.0))
        dominant_rate = float(diversity.get("dominant_predicted_rate", 0.0))
        collapsed = bool(diversity.get("collapsed", False))
        if target_unique < 2:
            continue
        multi_target_profiles += 1
        max_dominant_rate = max(max_dominant_rate, dominant_rate)
        min_target_token_coverage = min(min_target_token_coverage, target_token_coverage)
        passed = (
            not collapsed
            and predicted_unique >= target_unique
            and target_token_coverage >= 1.0
        )
        if passed:
            passed_profiles += 1
            continue
        blocking_evals.append(
            {
                "name": name,
                "target_unique": target_unique,
                "predicted_unique": predicted_unique,
                "target_token_coverage": target_token_coverage,
                "dominant_predicted_token": diversity.get("dominant_predicted_token"),
                "dominant_predicted_rate": dominant_rate,
                "collapsed": collapsed,
                "missing_target_tokens": diversity.get("missing_target_tokens", []),
            }
        )

    return {
        "passed": multi_target_profiles > 0 and passed_profiles == multi_target_profiles,
        "multi_target_profiles": multi_target_profiles,
        "passed_profiles": passed_profiles,
        "failed_profiles": len(blocking_evals),
        "max_dominant_predicted_rate": (
            max_dominant_rate
            if multi_target_profiles
            else 0.0
        ),
        "min_target_token_coverage": (
            min_target_token_coverage
            if multi_target_profiles
            else 0.0
        ),
        "blocking_evals": blocking_evals,
        "root_cause": branch_diversity_root_cause_summary(branch_profiles),
    }
