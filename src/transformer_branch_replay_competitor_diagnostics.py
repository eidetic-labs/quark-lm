"""Dominant competitor diagnostics for branch replay records."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from replay_plan import BranchReplayRecord, branch_replay_parts


def branch_replay_competitor_summary(
    model: Any,
    tokenizer: Any,
    branches: list[BranchReplayRecord],
) -> dict[str, Any]:
    """Summarize top competing predictions for a branch-replay surface."""

    records = [_competitor_record(model, tokenizer, branch) for branch in branches]
    return {
        **_competitor_summary(records),
        "profiles": {
            profile: _competitor_summary(profile_records)
            for profile, profile_records in sorted(
                _records_by_profile(records).items()
            )
        },
    }


def branch_replay_competitor_movement(
    step_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize first-to-last competitor movement across repair steps."""

    summaries = [
        record.get("target_floor_competitor_summary")
        for record in step_records
        if isinstance(record.get("target_floor_competitor_summary"), dict)
    ]
    if not summaries:
        return {"available": False}
    first = summaries[0]
    last = summaries[-1]
    first_win = float(first.get("target_won_rate", 0.0))
    last_win = float(last.get("target_won_rate", 0.0))
    first_competitor = float(first.get("dominant_competitor_rate", 0.0))
    last_competitor = float(last.get("dominant_competitor_rate", 0.0))
    first_margin = float(first.get("avg_losing_margin", 0.0))
    last_margin = float(last.get("avg_losing_margin", 0.0))
    return {
        "available": True,
        "first_target_won_rate": first_win,
        "last_target_won_rate": last_win,
        "target_won_rate_delta": last_win - first_win,
        "first_dominant_competitor_rate": first_competitor,
        "last_dominant_competitor_rate": last_competitor,
        "dominant_competitor_rate_delta": last_competitor - first_competitor,
        "first_avg_losing_margin": first_margin,
        "last_avg_losing_margin": last_margin,
        "avg_losing_margin_delta": last_margin - first_margin,
        "target_won_rate_improved": last_win > first_win,
        "dominant_competitor_rate_reduced": last_competitor < first_competitor,
        "losing_margin_reduced": last_margin < first_margin,
    }


def _competitor_record(
    model: Any,
    tokenizer: Any,
    branch: BranchReplayRecord,
) -> dict[str, Any]:
    context, target, _predicted, profile = branch_replay_parts(branch)
    probs = model.predict(context)
    ranked_ids = sorted(
        range(len(probs)),
        key=lambda index: (-probs[index], _token_value(tokenizer, index), index),
    )
    top_id = ranked_ids[0]
    target_prob = float(probs[target])
    top_prob = float(probs[top_id])
    return {
        "profile": profile,
        "target": target,
        "target_token": _token_value(tokenizer, target),
        "top": top_id,
        "top_token": _token_value(tokenizer, top_id),
        "target_won": top_id == target,
        "top_vs_target_margin": top_prob - target_prob,
    }


def _competitor_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return _empty_summary()
    count = len(records)
    target_won_count = sum(1 for record in records if bool(record["target_won"]))
    losing_records = [record for record in records if not bool(record["target_won"])]
    top_counts = Counter(int(record["top"]) for record in records)
    losing_top_counts = Counter(int(record["top"]) for record in losing_records)
    dominant_id, dominant_count = _most_common(losing_top_counts)
    return {
        "count": count,
        "target_won_count": target_won_count,
        "target_won_rate": target_won_count / count,
        "competitor_count": len(losing_records),
        "dominant_competitor_id": dominant_id,
        "dominant_competitor_token": _record_token(records, dominant_id),
        "dominant_competitor_count": dominant_count,
        "dominant_competitor_rate": dominant_count / count if count else 0.0,
        "avg_top_vs_target_margin": _avg_margin(records),
        "avg_losing_margin": _avg_margin(losing_records),
        "top_tokens": _token_items(records, top_counts, count),
        "competitor_tokens": _token_items(records, losing_top_counts, count),
    }


def _empty_summary() -> dict[str, Any]:
    return {
        "count": 0,
        "target_won_count": 0,
        "target_won_rate": 0.0,
        "competitor_count": 0,
        "dominant_competitor_id": None,
        "dominant_competitor_token": None,
        "dominant_competitor_count": 0,
        "dominant_competitor_rate": 0.0,
        "avg_top_vs_target_margin": 0.0,
        "avg_losing_margin": 0.0,
        "top_tokens": [],
        "competitor_tokens": [],
    }


def _records_by_profile(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_profile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_profile[str(record["profile"])].append(record)
    return dict(by_profile)


def _most_common(counter: Counter[int]) -> tuple[int | None, int]:
    if not counter:
        return None, 0
    return counter.most_common(1)[0]


def _avg_margin(records: list[dict[str, Any]]) -> float:
    if not records:
        return 0.0
    return sum(float(record["top_vs_target_margin"]) for record in records) / len(
        records
    )


def _token_items(
    records: list[dict[str, Any]],
    counts: Counter[int],
    total_count: int,
) -> list[dict[str, Any]]:
    return [
        {
            "id": token_id,
            "value": _record_token(records, token_id),
            "count": count,
            "rate": count / total_count if total_count else 0.0,
        }
        for token_id, count in counts.most_common(8)
    ]


def _record_token(records: list[dict[str, Any]], token_id: int | None) -> str | None:
    if token_id is None:
        return None
    for record in records:
        if int(record["top"]) == token_id:
            return str(record["top_token"])
        if int(record["target"]) == token_id:
            return str(record["target_token"])
    return str(token_id)


def _token_value(tokenizer: Any, token_id: int) -> str:
    tokens = getattr(tokenizer, "itos", None)
    if isinstance(tokens, list) and 0 <= token_id < len(tokens):
        return str(tokens[token_id])
    if isinstance(tokens, dict) and token_id in tokens:
        return str(tokens[token_id])
    return str(token_id)
