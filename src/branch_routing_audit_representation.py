"""Representation-separation audit for branch routing."""

from __future__ import annotations

from typing import Any

from branch_diversity_diagnostics import _as_float, _as_int


def branch_representation_audit_summary(
    branch_representation_profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
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
            "target_centroid_distance_min": _as_float(
                profile.get("target_centroid_distance", {}).get("min")
            ),
            "target_centroid_distance_avg": _as_float(
                profile.get("target_centroid_distance", {}).get("avg")
            ),
            "target_centroid_margin_min": _as_float(
                profile.get("target_centroid_margin", {}).get("min")
            ),
            "target_centroid_margin_avg": _as_float(
                profile.get("target_centroid_margin", {}).get("avg")
            ),
            "poorly_separated_centroid_rate": _as_float(
                profile.get("target_centroid_margin", {}).get(
                    "poorly_separated_rate"
                )
            ),
        }
        representation_profiles.append(record)
        different_distances.append(different_avg)
        if (
            different_avg <= 0.01
            or separation_ratio <= 1.05
            or record["target_centroid_distance_min"] <= 0.01
            or record["target_centroid_margin_min"] <= 0.01
        ):
            low_separation_profiles.append(record)

    return {
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
    }

