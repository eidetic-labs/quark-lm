"""Compact gap summaries for direct-answer frontier comparisons."""

from __future__ import annotations

from typing import Any


def build_frontier_gap_summary(comparison: Any) -> dict[str, Any]:
    """Summarize residual frontier gaps without turning them into training data."""

    if not isinstance(comparison, dict):
        return _inactive_summary("comparison_missing")
    if comparison.get("available") is not True:
        return _inactive_summary(str(comparison.get("reason", "unavailable")))

    coverage_delta = _as_dict(comparison.get("coverage_delta"))
    profile_diagnostics = _as_dict(
        comparison.get("profile_regression_diagnostics")
    )
    regressed_profiles = _profile_names(coverage_delta.get("regressed_profiles"))
    return {
        "active": True,
        "available": True,
        "passed": comparison.get("passed") is True,
        "used_for_training": False,
        "rule": (
            "Frontier gaps are repair-focus evidence only; they are not "
            "training examples, weights, tokenizer state, or embeddings."
        ),
        "coverage_preserved": comparison.get("coverage_preserved") is True,
        "stability_preserved": comparison.get("stability_preserved") is True,
        "score_preserved": comparison.get("score_preserved") is True,
        "coverage_regressed_profile_count": len(regressed_profiles),
        "coverage_regressed_profiles": regressed_profiles,
        "improved_profile_count": int(coverage_delta.get("improved_profile_count", 0)),
        "improved_profiles": _profile_names(coverage_delta.get("improved_profiles")),
        "tied_profile_count": int(coverage_delta.get("tied_profile_count", 0)),
        "tied_profiles": list(coverage_delta.get("tied_profiles", [])),
        "diagnosis_label_counts": dict(
            sorted(_as_dict(profile_diagnostics.get("diagnosis_label_counts")).items())
        ),
        "worst_profile": _worst_profile_name(profile_diagnostics),
        "repair_focus_profiles": _repair_focus_profiles(
            regressed_profiles,
            profile_diagnostics,
        ),
    }


def _inactive_summary(reason: str) -> dict[str, Any]:
    return {
        "active": False,
        "available": False,
        "passed": False,
        "used_for_training": False,
        "reason": reason,
        "repair_focus_profiles": [],
    }


def _repair_focus_profiles(
    regressed_profiles: list[str],
    profile_diagnostics: dict[str, Any],
) -> list[str]:
    diagnostic_profiles = _profile_names(profile_diagnostics.get("profiles"))
    return sorted(set(regressed_profiles) | set(diagnostic_profiles))


def _profile_names(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    names = []
    for item in value:
        if isinstance(item, dict):
            name = str(item.get("profile", ""))
            if name:
                names.append(name)
        elif isinstance(item, str):
            names.append(item)
    return sorted(set(names))


def _worst_profile_name(profile_diagnostics: dict[str, Any]) -> str | None:
    worst = profile_diagnostics.get("worst_profile")
    if not isinstance(worst, dict):
        return None
    name = str(worst.get("profile", ""))
    return name or None


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
