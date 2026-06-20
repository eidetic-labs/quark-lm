"""Profile-level diagnostics for branch frontier regressions."""

from __future__ import annotations

from collections import Counter
from typing import Any

from branch_diversity_snapshot_coverage import (
    branch_diversity_snapshot_target_coverage_delta,
)
from transformer_branch_frontier_profile_sources import (
    logit_prior_summary,
    snapshot_branch_profile,
    snapshot_logit_prior_profile,
    snapshot_representation_profile,
)


def branch_frontier_profile_regression_diagnostics(
    *,
    snapshot: dict[str, Any],
    frontier_snapshot: dict[str, Any],
) -> dict[str, Any]:
    coverage_delta = branch_diversity_snapshot_target_coverage_delta(
        snapshot,
        frontier_snapshot,
    )
    profiles = [
        _profile_diagnostic(
            profile_name=str(item["profile"]),
            coverage_delta=item,
            snapshot=snapshot,
            frontier_snapshot=frontier_snapshot,
        )
        for item in coverage_delta["regressed_profiles"]
    ]
    profiles.sort(
        key=lambda item: (
            float(item["coverage"]["delta"]),
            -float(item["dominant_prediction"]["dominant_rate_delta"]),
            str(item["profile"]),
        )
    )
    labels = Counter(
        label for profile in profiles for label in profile["diagnosis_labels"]
    )
    return {
        "regressed_profile_count": len(profiles),
        "zero_coverage_regression_count": int(labels["zero_coverage_regression"]),
        "prediction_collapse_regression_count": int(
            labels["prediction_collapse_regression"]
        ),
        "target_rank_regression_count": int(labels["target_rank_regression"]),
        "target_topk_regression_count": int(labels["target_topk_regression"]),
        "hidden_pressure_regression_count": int(
            labels["hidden_projection_pressure_regression"]
        ),
        "representation_margin_regression_count": int(
            labels["representation_margin_regression"]
        ),
        "diagnosis_label_counts": dict(sorted(labels.items())),
        "worst_profile": profiles[0] if profiles else None,
        "profiles": profiles,
    }


def _profile_diagnostic(
    *,
    profile_name: str,
    coverage_delta: dict[str, Any],
    snapshot: dict[str, Any],
    frontier_snapshot: dict[str, Any],
) -> dict[str, Any]:
    snapshot_profile = snapshot_branch_profile(snapshot, profile_name)
    frontier_profile = snapshot_branch_profile(frontier_snapshot, profile_name)
    prediction = _dominant_prediction_delta(snapshot_profile, frontier_profile)
    rank = _target_rank_delta(snapshot_profile, frontier_profile)
    representation = _representation_delta(
        snapshot_representation_profile(snapshot, profile_name),
        snapshot_representation_profile(frontier_snapshot, profile_name),
    )
    logit = _logit_prior_delta(
        snapshot_logit_prior_profile(snapshot, profile_name),
        snapshot_logit_prior_profile(frontier_snapshot, profile_name),
    )
    labels = _diagnosis_labels(
        coverage_delta=coverage_delta,
        prediction=prediction,
        rank=rank,
        representation=representation,
        logit=logit,
    )
    return {
        "profile": profile_name,
        "coverage": {
            "frontier": float(coverage_delta.get("baseline_coverage", 0.0)),
            "snapshot": float(coverage_delta.get("snapshot_coverage", 0.0)),
            "delta": float(coverage_delta.get("delta", 0.0)),
        },
        "dominant_prediction": prediction,
        "target_rank": rank,
        "representation": representation,
        "logit_prior": logit,
        "diagnosis_labels": labels,
    }


def _diagnosis_labels(
    *,
    coverage_delta: dict[str, Any],
    prediction: dict[str, Any],
    rank: dict[str, Any],
    representation: dict[str, Any],
    logit: dict[str, Any],
) -> list[str]:
    labels: list[str] = []
    if (
        float(coverage_delta.get("snapshot_coverage", 0.0)) <= 1e-12
        and float(coverage_delta.get("baseline_coverage", 0.0)) > 1e-12
    ):
        labels.append("zero_coverage_regression")
    if bool(prediction["became_collapsed"]):
        labels.append("prediction_collapse_regression")
    if int(prediction["predicted_unique_delta"]) < 0:
        labels.append("prediction_diversity_regression")
    if float(prediction["dominant_rate_delta"]) > 1e-12:
        labels.append("dominant_prediction_regression")
    if float(rank["avg_delta"]) > 1e-12:
        labels.append("target_rank_regression")
    if float(rank["top3_delta"]) < -1e-12 or float(rank["top5_delta"]) < -1e-12:
        labels.append("target_topk_regression")
    if float(representation["target_centroid_margin_min_delta"]) < -1e-12:
        labels.append("representation_margin_regression")
    if float(logit["avg_hidden_advantage_delta"]) > 1e-12:
        labels.append("hidden_projection_pressure_regression")
    return labels


def _dominant_prediction_delta(
    snapshot_profile: dict[str, Any],
    frontier_profile: dict[str, Any],
) -> dict[str, Any]:
    snapshot_diversity = _as_dict(snapshot_profile.get("diversity"))
    frontier_diversity = _as_dict(frontier_profile.get("diversity"))
    snapshot_unique = int(snapshot_diversity.get("predicted_unique", 0))
    frontier_unique = int(frontier_diversity.get("predicted_unique", 0))
    snapshot_rate = float(snapshot_diversity.get("dominant_predicted_rate", 0.0))
    frontier_rate = float(frontier_diversity.get("dominant_predicted_rate", 0.0))
    snapshot_collapsed = bool(snapshot_diversity.get("collapsed", False))
    frontier_collapsed = bool(frontier_diversity.get("collapsed", False))
    return {
        "frontier_token": frontier_diversity.get("dominant_predicted_token"),
        "snapshot_token": snapshot_diversity.get("dominant_predicted_token"),
        "token_changed": snapshot_diversity.get("dominant_predicted_token")
        != frontier_diversity.get("dominant_predicted_token"),
        "frontier_rate": frontier_rate,
        "snapshot_rate": snapshot_rate,
        "dominant_rate_delta": snapshot_rate - frontier_rate,
        "frontier_predicted_unique": frontier_unique,
        "snapshot_predicted_unique": snapshot_unique,
        "predicted_unique_delta": snapshot_unique - frontier_unique,
        "frontier_collapsed": frontier_collapsed,
        "snapshot_collapsed": snapshot_collapsed,
        "became_collapsed": snapshot_collapsed and not frontier_collapsed,
    }


def _target_rank_delta(
    snapshot_profile: dict[str, Any],
    frontier_profile: dict[str, Any],
) -> dict[str, float]:
    snapshot_rank = _as_dict(snapshot_profile.get("target_rank"))
    frontier_rank = _as_dict(frontier_profile.get("target_rank"))
    snapshot_avg = float(snapshot_rank.get("avg", 0.0))
    frontier_avg = float(frontier_rank.get("avg", 0.0))
    snapshot_top3 = float(snapshot_rank.get("top3_rate", 0.0))
    frontier_top3 = float(frontier_rank.get("top3_rate", 0.0))
    snapshot_top5 = float(snapshot_rank.get("top5_rate", 0.0))
    frontier_top5 = float(frontier_rank.get("top5_rate", 0.0))
    return {
        "frontier_avg": frontier_avg,
        "snapshot_avg": snapshot_avg,
        "avg_delta": snapshot_avg - frontier_avg,
        "frontier_top3_rate": frontier_top3,
        "snapshot_top3_rate": snapshot_top3,
        "top3_delta": snapshot_top3 - frontier_top3,
        "frontier_top5_rate": frontier_top5,
        "snapshot_top5_rate": snapshot_top5,
        "top5_delta": snapshot_top5 - frontier_top5,
    }


def _representation_delta(
    snapshot_profile: dict[str, Any],
    frontier_profile: dict[str, Any],
) -> dict[str, float]:
    snapshot_margin = _as_dict(snapshot_profile.get("target_centroid_margin"))
    frontier_margin = _as_dict(frontier_profile.get("target_centroid_margin"))
    snapshot_distance = _as_dict(snapshot_profile.get("different_target_pairwise_distance"))
    frontier_distance = _as_dict(frontier_profile.get("different_target_pairwise_distance"))
    snapshot_margin_min = float(snapshot_margin.get("min", 0.0))
    frontier_margin_min = float(frontier_margin.get("min", 0.0))
    snapshot_poor_rate = float(snapshot_margin.get("poorly_separated_rate", 0.0))
    frontier_poor_rate = float(frontier_margin.get("poorly_separated_rate", 0.0))
    snapshot_distance_avg = float(snapshot_distance.get("avg", 0.0))
    frontier_distance_avg = float(frontier_distance.get("avg", 0.0))
    return {
        "frontier_target_centroid_margin_min": frontier_margin_min,
        "snapshot_target_centroid_margin_min": snapshot_margin_min,
        "target_centroid_margin_min_delta": snapshot_margin_min - frontier_margin_min,
        "frontier_poorly_separated_rate": frontier_poor_rate,
        "snapshot_poorly_separated_rate": snapshot_poor_rate,
        "poorly_separated_rate_delta": snapshot_poor_rate - frontier_poor_rate,
        "frontier_different_target_distance_avg": frontier_distance_avg,
        "snapshot_different_target_distance_avg": snapshot_distance_avg,
        "different_target_distance_avg_delta": (
            snapshot_distance_avg - frontier_distance_avg
        ),
    }


def _logit_prior_delta(
    snapshot_profile: dict[str, Any],
    frontier_profile: dict[str, Any],
) -> dict[str, Any]:
    snapshot_summary = logit_prior_summary(snapshot_profile)
    frontier_summary = logit_prior_summary(frontier_profile)
    snapshot_hidden = float(snapshot_summary.get("avg_hidden_advantage", 0.0))
    frontier_hidden = float(frontier_summary.get("avg_hidden_advantage", 0.0))
    snapshot_win_rate = float(snapshot_summary.get("dominant_logit_win_rate", 0.0))
    frontier_win_rate = float(frontier_summary.get("dominant_logit_win_rate", 0.0))
    return {
        "frontier_primary_pressure": frontier_summary.get("primary_pressure"),
        "snapshot_primary_pressure": snapshot_summary.get("primary_pressure"),
        "frontier_avg_hidden_advantage": frontier_hidden,
        "snapshot_avg_hidden_advantage": snapshot_hidden,
        "avg_hidden_advantage_delta": snapshot_hidden - frontier_hidden,
        "frontier_dominant_logit_win_rate": frontier_win_rate,
        "snapshot_dominant_logit_win_rate": snapshot_win_rate,
        "dominant_logit_win_rate_delta": snapshot_win_rate - frontier_win_rate,
    }


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
