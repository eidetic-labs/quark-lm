"""Rank diagnostics for branch replay records."""

from __future__ import annotations

from collections import Counter
from typing import Any

from replay_plan import BranchReplayRecord, branch_replay_parts


def branch_replay_rank_summary(
    model: Any,
    branches: list[BranchReplayRecord],
) -> dict[str, Any]:
    """Summarize target rank for a branch-replay surface."""

    summaries = [_rank_record(model, branch) for branch in branches]
    return {
        **_rank_summary(summaries),
        "profiles": {
            profile: _rank_summary(profile_summaries)
            for profile, profile_summaries in sorted(
                _summaries_by_profile(summaries).items()
            )
        },
    }


def branch_replay_rank_movement(step_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize first-to-last rank movement across routing-repair steps."""

    summaries = [
        record.get("target_floor_rank_summary")
        for record in step_records
        if isinstance(record.get("target_floor_rank_summary"), dict)
    ]
    if not summaries:
        return {"available": False}
    first = summaries[0]
    last = summaries[-1]
    first_avg = float(first.get("avg_target_rank", 0.0))
    last_avg = float(last.get("avg_target_rank", 0.0))
    first_top1 = float(first.get("top1_rate", 0.0))
    last_top1 = float(last.get("top1_rate", 0.0))
    return {
        "available": True,
        "first_avg_target_rank": first_avg,
        "last_avg_target_rank": last_avg,
        "avg_target_rank_delta": last_avg - first_avg,
        "first_top1_rate": first_top1,
        "last_top1_rate": last_top1,
        "top1_rate_delta": last_top1 - first_top1,
        "rank_improved": last_avg < first_avg,
        "top1_improved": last_top1 > first_top1,
    }


def _rank_record(model: Any, branch: BranchReplayRecord) -> dict[str, Any]:
    context, target, predicted, profile = branch_replay_parts(branch)
    probs = model.predict(context)
    ranked_ids = sorted(range(len(probs)), key=lambda index: (-probs[index], index))
    target_rank = ranked_ids.index(target) + 1
    target_prob = float(probs[target])
    strongest_non_target = max(
        (float(prob) for index, prob in enumerate(probs) if index != target),
        default=0.0,
    )
    return {
        "profile": profile,
        "target": target,
        "predicted": predicted,
        "target_rank": target_rank,
        "target_prob": target_prob,
        "target_margin": target_prob - strongest_non_target,
    }


def _rank_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return _empty_summary()
    target_counts = Counter(int(record["target"]) for record in records)
    target_ranks = [int(record["target_rank"]) for record in records]
    target_probs = [float(record["target_prob"]) for record in records]
    target_margins = [float(record["target_margin"]) for record in records]
    count = len(records)
    return {
        "count": count,
        "target_count": len(target_counts),
        "avg_target_rank": sum(target_ranks) / count,
        "best_target_rank": min(target_ranks),
        "worst_target_rank": max(target_ranks),
        "top1_rate": sum(rank <= 1 for rank in target_ranks) / count,
        "top3_rate": sum(rank <= 3 for rank in target_ranks) / count,
        "top5_rate": sum(rank <= 5 for rank in target_ranks) / count,
        "avg_target_prob": sum(target_probs) / count,
        "avg_target_margin": sum(target_margins) / count,
    }


def _empty_summary() -> dict[str, Any]:
    return {
        "count": 0,
        "target_count": 0,
        "avg_target_rank": 0.0,
        "best_target_rank": 0,
        "worst_target_rank": 0,
        "top1_rate": 0.0,
        "top3_rate": 0.0,
        "top5_rate": 0.0,
        "avg_target_prob": 0.0,
        "avg_target_margin": 0.0,
    }


def _summaries_by_profile(
    summaries: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_profile: dict[str, list[dict[str, Any]]] = {}
    for summary in summaries:
        by_profile.setdefault(str(summary["profile"]), []).append(summary)
    return by_profile
