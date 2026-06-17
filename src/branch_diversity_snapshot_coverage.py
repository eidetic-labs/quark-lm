"""Target-coverage diagnostics for branch-diversity snapshots."""

from __future__ import annotations

from typing import Any


def branch_diversity_snapshot_target_coverage_by_profile(
    snapshot: dict[str, Any],
) -> dict[str, float]:
    coverage_by_profile: dict[str, float] = {}
    branch_profiles = snapshot.get("branch_profiles", {})
    for name, profile in branch_profiles.items():
        diversity = profile.get("diversity", {})
        target_unique = int(diversity.get("target_unique", 0))
        if target_unique < 2:
            continue
        coverage_by_profile[name] = float(
            diversity.get("target_token_coverage", 0.0)
        )
    return coverage_by_profile


def branch_diversity_snapshot_target_coverage_diagnostics(
    snapshot: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    baseline_coverage = branch_diversity_snapshot_target_coverage_by_profile(baseline)
    snapshot_coverage = branch_diversity_snapshot_target_coverage_by_profile(snapshot)
    violations: list[dict[str, Any]] = []
    for name, baseline_value in sorted(baseline_coverage.items()):
        snapshot_value = float(snapshot_coverage.get(name, -1.0))
        deficit = float(baseline_value - snapshot_value)
        if snapshot_value + 1e-12 < baseline_value:
            violations.append(
                {
                    "profile": name,
                    "baseline_coverage": float(baseline_value),
                    "snapshot_coverage": snapshot_value,
                    "deficit": deficit,
                }
            )
    violations.sort(key=lambda item: (-float(item["deficit"]), item["profile"]))
    worst_violation = violations[0] if violations else None
    return {
        "preserved": not violations,
        "baseline_profile_count": len(baseline_coverage),
        "snapshot_profile_count": len(snapshot_coverage),
        "violating_profile_count": len(violations),
        "worst_deficit": float(worst_violation["deficit"]) if worst_violation else 0.0,
        "worst_violation": worst_violation,
        "violations": violations,
    }


def branch_diversity_snapshot_target_coverage_delta(
    snapshot: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    baseline_coverage = branch_diversity_snapshot_target_coverage_by_profile(baseline)
    snapshot_coverage = branch_diversity_snapshot_target_coverage_by_profile(snapshot)
    profile_names = sorted(set(baseline_coverage) | set(snapshot_coverage))
    improved_profiles: list[dict[str, Any]] = []
    regressed_profiles: list[dict[str, Any]] = []
    tied_profiles: list[str] = []
    deltas: dict[str, float] = {}
    for name in profile_names:
        baseline_value = float(baseline_coverage.get(name, 0.0))
        snapshot_value = float(snapshot_coverage.get(name, 0.0))
        delta = snapshot_value - baseline_value
        deltas[name] = delta
        profile_delta = {
            "profile": name,
            "baseline_coverage": baseline_value,
            "snapshot_coverage": snapshot_value,
            "delta": delta,
        }
        if delta > 1e-12:
            improved_profiles.append(profile_delta)
        elif delta < -1e-12:
            regressed_profiles.append(profile_delta)
        else:
            tied_profiles.append(name)
    profile_count = max(len(profile_names), 1)
    baseline_average = sum(float(value) for value in baseline_coverage.values()) / max(
        len(baseline_coverage),
        1,
    )
    snapshot_average = sum(float(value) for value in snapshot_coverage.values()) / max(
        len(snapshot_coverage),
        1,
    )
    baseline_min = (
        min(float(value) for value in baseline_coverage.values())
        if baseline_coverage
        else 0.0
    )
    snapshot_min = (
        min(float(value) for value in snapshot_coverage.values())
        if snapshot_coverage
        else 0.0
    )
    return {
        "baseline_profile_count": len(baseline_coverage),
        "snapshot_profile_count": len(snapshot_coverage),
        "profile_count": len(profile_names),
        "baseline_min_coverage": baseline_min,
        "snapshot_min_coverage": snapshot_min,
        "min_delta": snapshot_min - baseline_min,
        "baseline_average_coverage": baseline_average,
        "snapshot_average_coverage": snapshot_average,
        "average_delta": snapshot_average - baseline_average,
        "total_delta": sum(deltas.values()),
        "mean_delta": sum(deltas.values()) / profile_count,
        "improved_profile_count": len(improved_profiles),
        "regressed_profile_count": len(regressed_profiles),
        "tied_profile_count": len(tied_profiles),
        "improved_profiles": improved_profiles,
        "regressed_profiles": regressed_profiles,
        "tied_profiles": tied_profiles,
        "deltas": deltas,
    }


def branch_diversity_snapshot_preserves_target_coverage(
    snapshot: dict[str, Any],
    baseline: dict[str, Any],
) -> bool:
    diagnostics = branch_diversity_snapshot_target_coverage_diagnostics(
        snapshot,
        baseline,
    )
    return bool(diagnostics["preserved"])
