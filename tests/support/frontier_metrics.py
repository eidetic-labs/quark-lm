"""Synthetic frontier metric fixtures for answer-sweep tests."""

from __future__ import annotations


def metrics_with_profile_coverage(
    run_id: str,
    coverage_by_profile: dict[str, float],
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "direct_answer": {
            "final": {
                "branch_profiles": {
                    profile: _profile_metrics(
                        coverage=coverage,
                        dominant_rate=1.0 - coverage,
                        top3_rate=coverage,
                        top5_rate=coverage,
                        avg_rank=2.0,
                    )
                    for profile, coverage in coverage_by_profile.items()
                },
                "branch_target_coverage_by_profile": coverage_by_profile,
                "branch_diversity_target": _branch_target_summary(
                    coverage_by_profile,
                ),
            }
        },
    }


def metrics_with_profile(
    run_id: str,
    *,
    coverage: float,
    dominant_rate: float,
    top3_rate: float,
    top5_rate: float,
    avg_rank: float,
) -> dict[str, object]:
    coverage_by_profile = {"qa": coverage}
    return {
        "run_id": run_id,
        "direct_answer": {
            "final": {
                "branch_profiles": {
                    "qa": _profile_metrics(
                        coverage=coverage,
                        dominant_rate=dominant_rate,
                        top3_rate=top3_rate,
                        top5_rate=top5_rate,
                        avg_rank=avg_rank,
                    )
                },
                "branch_target_coverage_by_profile": coverage_by_profile,
                "branch_diversity_target": _branch_target_summary(
                    coverage_by_profile,
                ),
            }
        },
    }


def _profile_metrics(
    *,
    coverage: float,
    dominant_rate: float,
    top3_rate: float,
    top5_rate: float,
    avg_rank: float,
) -> dict[str, object]:
    return {
        "diversity": {
            "target_unique": 2,
            "predicted_unique": 1,
            "target_token_coverage": coverage,
            "dominant_predicted_rate": dominant_rate,
            "collapsed": False,
        },
        "target_rank": {
            "top3_rate": top3_rate,
            "top5_rate": top5_rate,
            "avg": avg_rank,
        },
    }


def _branch_target_summary(
    coverage_by_profile: dict[str, float],
) -> dict[str, object]:
    return {
        "passed": False,
        "failed_profiles": len(coverage_by_profile),
        "passed_profiles": 0,
        "min_target_token_coverage": min(coverage_by_profile.values()),
        "root_cause": {"mode_counts": {"target_coverage_gap": 1}},
    }
