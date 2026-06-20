"""Top-1 collapse response diagnostics for branch-diversity snapshots."""

from __future__ import annotations

from typing import Any


def branch_diversity_collapse_response_delta(
    snapshot: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """Return profile-level top-1 collapse movement between snapshots."""

    baseline_profiles = _as_dict(baseline.get("branch_profiles"))
    snapshot_profiles = _as_dict(snapshot.get("branch_profiles"))
    improved_profiles = []
    regressed_profiles = []
    tied_profiles = []
    profiles = []

    for name in sorted(set(baseline_profiles) | set(snapshot_profiles)):
        baseline_profile = _profile_collapse_state(baseline_profiles.get(name))
        snapshot_profile = _profile_collapse_state(snapshot_profiles.get(name))
        if baseline_profile["target_unique"] < 2 and snapshot_profile["target_unique"] < 2:
            continue
        delta = _profile_delta(name, baseline_profile, snapshot_profile)
        profiles.append(delta)
        if _profile_improved(delta):
            improved_profiles.append(delta)
        elif _profile_regressed(delta):
            regressed_profiles.append(delta)
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


def branch_diversity_collapse_response_improved(
    snapshot: dict[str, Any],
    baseline: dict[str, Any],
) -> bool:
    """Return true when top-1 collapse improves without collapse regression."""

    delta = branch_diversity_collapse_response_delta(snapshot, baseline)
    return (
        int(delta["improved_profile_count"]) > 0
        and int(delta["regressed_profile_count"]) == 0
    )


def _profile_collapse_state(profile: Any) -> dict[str, Any]:
    diversity = _as_dict(_as_dict(profile).get("diversity"))
    predicted_unique = _as_int(diversity.get("predicted_unique"))
    target_unique = _as_int(diversity.get("target_unique"))
    dominant_rate = _as_float(diversity.get("dominant_predicted_rate"))
    collapsed = diversity.get("collapsed")
    return {
        "target_unique": target_unique,
        "predicted_unique": predicted_unique,
        "dominant_predicted_rate": dominant_rate,
        "collapsed": (
            bool(collapsed)
            if isinstance(collapsed, bool)
            else _is_collapsed(predicted_unique, target_unique, dominant_rate)
        ),
    }


def _profile_delta(
    name: str,
    baseline: dict[str, Any],
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    return {
        "profile": name,
        "baseline_predicted_unique": baseline["predicted_unique"],
        "snapshot_predicted_unique": snapshot["predicted_unique"],
        "predicted_unique_delta": (
            snapshot["predicted_unique"] - baseline["predicted_unique"]
        ),
        "baseline_dominant_rate": baseline["dominant_predicted_rate"],
        "snapshot_dominant_rate": snapshot["dominant_predicted_rate"],
        "dominant_rate_delta": (
            snapshot["dominant_predicted_rate"] - baseline["dominant_predicted_rate"]
        ),
        "baseline_collapsed": baseline["collapsed"],
        "snapshot_collapsed": snapshot["collapsed"],
    }


def _profile_improved(delta: dict[str, Any]) -> bool:
    return (
        int(delta["predicted_unique_delta"]) > 0
        or float(delta["dominant_rate_delta"]) < -1e-12
        or (
            bool(delta["baseline_collapsed"])
            and not bool(delta["snapshot_collapsed"])
        )
    ) and not _profile_regressed(delta)


def _profile_regressed(delta: dict[str, Any]) -> bool:
    return (
        int(delta["predicted_unique_delta"]) < 0
        or float(delta["dominant_rate_delta"]) > 1e-12
        or (
            not bool(delta["baseline_collapsed"])
            and bool(delta["snapshot_collapsed"])
        )
    )


def _is_collapsed(
    predicted_unique: int,
    target_unique: int,
    dominant_rate: float,
) -> bool:
    return target_unique > 1 and (predicted_unique <= 1 or dominant_rate >= 1.0)


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
