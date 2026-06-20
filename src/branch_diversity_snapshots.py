from __future__ import annotations

from typing import Any


def branch_diversity_snapshot_score(snapshot: dict[str, Any]) -> tuple[float, ...]:
    summary = snapshot.get("branch_diversity_target", {})
    branch_profiles = snapshot.get("branch_profiles", {})
    multi_target_diversities = []
    for profile in branch_profiles.values():
        diversity = profile.get("diversity", {})
        target_rank = profile.get("target_rank", {})
        target_unique = int(diversity.get("target_unique", 0))
        if target_unique < 2:
            continue
        predicted_unique = int(diversity.get("predicted_unique", 0))
        avg_target_rank = float(target_rank.get("avg", 0.0))
        multi_target_diversities.append(
            {
                "not_collapsed": 0.0
                if bool(diversity.get("collapsed", False))
                else 1.0,
                "predicted_unique_rate": predicted_unique / target_unique,
                "target_token_coverage": float(
                    diversity.get("target_token_coverage", 0.0)
                ),
                "inverse_dominant_rate": 1.0
                - float(diversity.get("dominant_predicted_rate", 0.0)),
                "target_top3_rate": float(target_rank.get("top3_rate", 0.0)),
                "target_top5_rate": float(target_rank.get("top5_rate", 0.0)),
                "inverse_target_rank": (
                    1.0 / avg_target_rank if avg_target_rank > 0.0 else 0.0
                ),
            }
        )
    profile_count = max(len(multi_target_diversities), 1)
    avg_predicted_unique_rate = (
        sum(item["predicted_unique_rate"] for item in multi_target_diversities)
        / profile_count
    )
    avg_not_collapsed_rate = (
        sum(item["not_collapsed"] for item in multi_target_diversities)
        / profile_count
    )
    avg_target_token_coverage = (
        sum(item["target_token_coverage"] for item in multi_target_diversities)
        / profile_count
    )
    avg_inverse_dominant_rate = (
        sum(item["inverse_dominant_rate"] for item in multi_target_diversities)
        / profile_count
    )
    avg_target_top3_rate = (
        sum(item["target_top3_rate"] for item in multi_target_diversities)
        / profile_count
    )
    avg_target_top5_rate = (
        sum(item["target_top5_rate"] for item in multi_target_diversities)
        / profile_count
    )
    avg_inverse_target_rank = (
        sum(item["inverse_target_rank"] for item in multi_target_diversities)
        / profile_count
    )
    return (
        1.0 if summary.get("passed", False) else 0.0,
        float(summary.get("passed_profiles", 0)),
        -float(summary.get("failed_profiles", 0)),
        float(summary.get("min_target_token_coverage", 0.0)),
        avg_not_collapsed_rate,
        avg_target_token_coverage,
        avg_target_top3_rate,
        avg_target_top5_rate,
        avg_inverse_target_rank,
        avg_predicted_unique_rate,
        avg_inverse_dominant_rate,
    )


def branch_diversity_snapshot_score_improved(
    snapshot: dict[str, Any],
    baseline: dict[str, Any],
) -> bool:
    return branch_diversity_snapshot_score(snapshot) > branch_diversity_snapshot_score(
        baseline
    )


def branch_diversity_snapshot_collapsed_profile_names(
    snapshot: dict[str, Any],
) -> list[str]:
    names: set[str] = set()
    for blocking_eval in snapshot.get("branch_diversity_target", {}).get(
        "blocking_evals",
        [],
    ):
        name = str(blocking_eval.get("name", ""))
        if not name:
            continue
        if bool(blocking_eval.get("collapsed", False)):
            names.add(name)
    return sorted(names)


__all__ = [
    "branch_diversity_profile_delta_has_coverage_gain",
    "branch_diversity_snapshot_collapsed_profile_names",
    "branch_diversity_snapshot_profile_diversity_delta",
    "branch_diversity_snapshot_score",
    "branch_diversity_snapshot_score_improved",
]


def branch_diversity_snapshot_profile_diversity_delta(
    snapshot: dict[str, Any],
    baseline: dict[str, Any],
    profile_names: list[str] | set[str] | tuple[str, ...],
) -> dict[str, Any]:
    baseline_profiles = baseline.get("branch_profiles", {})
    snapshot_profiles = snapshot.get("branch_profiles", {})
    improved_profiles: list[dict[str, Any]] = []
    regressed_profiles: list[dict[str, Any]] = []
    tied_profiles: list[str] = []
    profiles: list[dict[str, Any]] = []

    for name in sorted(set(profile_names)):
        baseline_diversity = baseline_profiles.get(name, {}).get("diversity", {})
        snapshot_diversity = snapshot_profiles.get(name, {}).get("diversity", {})
        baseline_predicted_unique = int(
            baseline_diversity.get("predicted_unique", 0)
        )
        snapshot_predicted_unique = int(
            snapshot_diversity.get("predicted_unique", 0)
        )
        baseline_coverage = float(
            baseline_diversity.get("target_token_coverage", 0.0)
        )
        snapshot_coverage = float(
            snapshot_diversity.get("target_token_coverage", 0.0)
        )
        baseline_dominant_rate = float(
            baseline_diversity.get("dominant_predicted_rate", 0.0)
        )
        snapshot_dominant_rate = float(
            snapshot_diversity.get("dominant_predicted_rate", 0.0)
        )
        predicted_unique_delta = (
            snapshot_predicted_unique - baseline_predicted_unique
        )
        coverage_delta = snapshot_coverage - baseline_coverage
        dominant_rate_delta = snapshot_dominant_rate - baseline_dominant_rate
        profile_delta = {
            "profile": name,
            "baseline_predicted_unique": baseline_predicted_unique,
            "snapshot_predicted_unique": snapshot_predicted_unique,
            "predicted_unique_delta": predicted_unique_delta,
            "baseline_coverage": baseline_coverage,
            "snapshot_coverage": snapshot_coverage,
            "coverage_delta": coverage_delta,
            "baseline_dominant_rate": baseline_dominant_rate,
            "snapshot_dominant_rate": snapshot_dominant_rate,
            "dominant_rate_delta": dominant_rate_delta,
        }
        profiles.append(profile_delta)
        improved = (
            predicted_unique_delta > 0
            or coverage_delta > 1e-12
            or dominant_rate_delta < -1e-12
        )
        regressed = predicted_unique_delta < 0 or coverage_delta < -1e-12
        if improved and not regressed:
            improved_profiles.append(profile_delta)
        elif regressed:
            regressed_profiles.append(profile_delta)
        else:
            tied_profiles.append(name)

    return {
        "profile_count": len(profiles),
        "improved_profile_count": len(improved_profiles),
        "regressed_profile_count": len(regressed_profiles),
        "tied_profile_count": len(tied_profiles),
        "improved_profiles": improved_profiles,
        "regressed_profiles": regressed_profiles,
        "tied_profiles": tied_profiles,
        "profiles": profiles,
    }


def branch_diversity_profile_delta_has_coverage_gain(delta: dict[str, Any]) -> bool:
    profiles = delta.get("profiles", [])
    if not isinstance(profiles, list):
        return False
    for profile in profiles:
        if not isinstance(profile, dict):
            continue
        if float(profile.get("coverage_delta", 0.0)) > 1e-12:
            return True
    return False
