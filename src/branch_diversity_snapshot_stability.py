"""Collapse-stability diagnostics for branch-diversity snapshots."""

from __future__ import annotations

from typing import Any

EPSILON = 1e-12


def branch_diversity_snapshot_stability_diagnostics(
    snapshot: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    baseline_profiles = _multi_target_profile_metrics(baseline)
    snapshot_profiles = _multi_target_profile_metrics(snapshot)
    violations: list[dict[str, Any]] = []
    for name, baseline_profile in sorted(baseline_profiles.items()):
        snapshot_profile = snapshot_profiles.get(name)
        if snapshot_profile is None:
            violations.append(_missing_profile_violation(name, baseline_profile))
            continue
        violations.extend(
            _profile_stability_violations(name, baseline_profile, snapshot_profile)
        )
    violations.sort(key=_violation_sort_key)
    violating_profiles = {str(violation["profile"]) for violation in violations}
    return {
        "preserved": not violations,
        "baseline_profile_count": len(baseline_profiles),
        "snapshot_profile_count": len(snapshot_profiles),
        "violating_profile_count": len(violating_profiles),
        "newly_collapsed_profile_count": _reason_count(
            violations,
            "newly_collapsed",
        ),
        "predicted_unique_regression_count": _reason_count(
            violations,
            "predicted_unique_regression",
        ),
        "dominant_rate_regression_count": _reason_count(
            violations,
            "dominant_rate_regression",
        ),
        "missing_profile_count": _reason_count(violations, "missing_profile"),
        "worst_violation": violations[0] if violations else None,
        "violations": violations,
    }


def branch_diversity_snapshot_preserves_profile_stability(
    snapshot: dict[str, Any],
    baseline: dict[str, Any],
) -> bool:
    diagnostics = branch_diversity_snapshot_stability_diagnostics(
        snapshot,
        baseline,
    )
    return bool(diagnostics["preserved"])


def _multi_target_profile_metrics(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    branch_profiles = snapshot.get("branch_profiles", {})
    if not isinstance(branch_profiles, dict):
        return metrics
    for name, profile in branch_profiles.items():
        if not isinstance(profile, dict):
            continue
        diversity = profile.get("diversity", {})
        if not isinstance(diversity, dict):
            continue
        target_unique = int(diversity.get("target_unique", 0))
        if target_unique < 2:
            continue
        predicted_unique = int(diversity.get("predicted_unique", 0))
        metrics[str(name)] = {
            "target_unique": target_unique,
            "predicted_unique": predicted_unique,
            "dominant_predicted_rate": float(
                diversity.get("dominant_predicted_rate", 0.0)
            ),
            "collapsed": _is_collapsed(diversity, target_unique, predicted_unique),
        }
    return metrics


def _is_collapsed(
    diversity: dict[str, Any],
    target_unique: int,
    predicted_unique: int,
) -> bool:
    if "collapsed" in diversity:
        return bool(diversity["collapsed"])
    return target_unique > 1 and predicted_unique <= 1


def _missing_profile_violation(
    name: str,
    baseline_profile: dict[str, Any],
) -> dict[str, Any]:
    return {
        "profile": name,
        "reason": "missing_profile",
        "baseline_predicted_unique": baseline_profile["predicted_unique"],
        "snapshot_predicted_unique": 0,
        "predicted_unique_delta": -int(baseline_profile["predicted_unique"]),
        "baseline_dominant_rate": baseline_profile["dominant_predicted_rate"],
        "snapshot_dominant_rate": 1.0,
        "dominant_rate_delta": (
            1.0 - float(baseline_profile["dominant_predicted_rate"])
        ),
        "baseline_collapsed": baseline_profile["collapsed"],
        "snapshot_collapsed": True,
    }


def _profile_stability_violations(
    name: str,
    baseline_profile: dict[str, Any],
    snapshot_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    if not baseline_profile["collapsed"] and snapshot_profile["collapsed"]:
        violations.append(
            _profile_violation(
                name,
                "newly_collapsed",
                baseline_profile,
                snapshot_profile,
            )
        )
    if int(snapshot_profile["predicted_unique"]) < int(
        baseline_profile["predicted_unique"]
    ):
        violations.append(
            _profile_violation(
                name,
                "predicted_unique_regression",
                baseline_profile,
                snapshot_profile,
            )
        )
    dominant_delta = float(snapshot_profile["dominant_predicted_rate"]) - float(
        baseline_profile["dominant_predicted_rate"]
    )
    if dominant_delta > EPSILON:
        violations.append(
            _profile_violation(
                name,
                "dominant_rate_regression",
                baseline_profile,
                snapshot_profile,
            )
        )
    return violations


def _profile_violation(
    name: str,
    reason: str,
    baseline_profile: dict[str, Any],
    snapshot_profile: dict[str, Any],
) -> dict[str, Any]:
    baseline_predicted_unique = int(baseline_profile["predicted_unique"])
    snapshot_predicted_unique = int(snapshot_profile["predicted_unique"])
    baseline_dominant_rate = float(baseline_profile["dominant_predicted_rate"])
    snapshot_dominant_rate = float(snapshot_profile["dominant_predicted_rate"])
    return {
        "profile": name,
        "reason": reason,
        "baseline_predicted_unique": baseline_predicted_unique,
        "snapshot_predicted_unique": snapshot_predicted_unique,
        "predicted_unique_delta": snapshot_predicted_unique
        - baseline_predicted_unique,
        "baseline_dominant_rate": baseline_dominant_rate,
        "snapshot_dominant_rate": snapshot_dominant_rate,
        "dominant_rate_delta": snapshot_dominant_rate - baseline_dominant_rate,
        "baseline_collapsed": bool(baseline_profile["collapsed"]),
        "snapshot_collapsed": bool(snapshot_profile["collapsed"]),
    }


def _reason_count(violations: list[dict[str, Any]], reason: str) -> int:
    return sum(1 for violation in violations if violation["reason"] == reason)


def _violation_sort_key(violation: dict[str, Any]) -> tuple[int, str]:
    order = {
        "missing_profile": 0,
        "newly_collapsed": 1,
        "predicted_unique_regression": 2,
        "dominant_rate_regression": 3,
    }
    return (order.get(str(violation["reason"]), 99), str(violation["profile"]))
