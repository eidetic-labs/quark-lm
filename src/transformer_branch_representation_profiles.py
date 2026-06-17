"""Branch hidden-representation diagnostics for direct-answer runs."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from answer_model import AnswerExample
from tokenizer import CharTokenizer
from transformer_direct_answer_core import direct_answer_branch_context
from transformer_direct_modes import ANSWER_TERMINATOR


def direct_answer_branch_representation_profile(
    model: Any,
    tokenizer: CharTokenizer,
    records: list[dict[str, Any]],
    branch_position: int,
    terminator: str = ANSWER_TERMINATOR,
) -> dict[str, Any]:
    representations: list[tuple[str, list[float]]] = []
    skipped = 0
    for record in records:
        example = AnswerExample(
            prompt=record["prompt"],
            target=record["target"],
            source=f"eval:{record['id']}",
        )
        branch = direct_answer_branch_context(
            model,
            tokenizer,
            example,
            branch_position,
            terminator,
        )
        if branch is None:
            skipped += 1
            continue
        context, target_id, _position = branch
        representations.append((tokenizer.itos[target_id], model.final_hidden(context)))

    all_distances: list[float] = []
    same_target_distances: list[float] = []
    different_target_distances: list[float] = []
    for left_index, (left_target, left_hidden) in enumerate(representations):
        for right_target, right_hidden in representations[left_index + 1:]:
            distance = _distance(left_hidden, right_hidden)
            all_distances.append(distance)
            if left_target == right_target:
                same_target_distances.append(distance)
            else:
                different_target_distances.append(distance)

    target_tokens = Counter(target for target, _hidden in representations)
    hidden_by_target: dict[str, list[list[float]]] = defaultdict(list)
    for target, hidden in representations:
        hidden_by_target[target].append(hidden)

    centroids = {
        target: _centroid(items)
        for target, items in sorted(hidden_by_target.items())
        if items
    }
    centroid_distances: list[float] = []
    centroid_items = list(centroids.items())
    for left_index, (_left_target, left_centroid) in enumerate(centroid_items):
        for _right_target, right_centroid in centroid_items[left_index + 1:]:
            centroid_distances.append(_distance(left_centroid, right_centroid))

    centroid_margins: list[float] = []
    for target, hidden in representations:
        own_centroid = centroids.get(target)
        other_centroids = [
            centroid_value
            for other_target, centroid_value in centroid_items
            if other_target != target
        ]
        if own_centroid is None or not other_centroids:
            continue
        own_distance = _distance(hidden, own_centroid)
        nearest_other_distance = min(
            _distance(hidden, other_centroid) for other_centroid in other_centroids
        )
        centroid_margins.append(nearest_other_distance - own_distance)

    return {
        "branch_position": branch_position,
        "count": len(representations),
        "skipped": skipped,
        "target_unique": len(target_tokens),
        "target_tokens": [
            {"value": value, "count": count}
            for value, count in target_tokens.most_common(12)
        ],
        "pairwise_distance": _summarize_distances(all_distances),
        "same_target_pairwise_distance": _summarize_distances(
            same_target_distances
        ),
        "different_target_pairwise_distance": _summarize_distances(
            different_target_distances
        ),
        "target_centroids": [
            {
                "target": target,
                "count": len(hidden_by_target[target]),
                "norm": math.sqrt(sum(value * value for value in centroid_value)),
            }
            for target, centroid_value in centroid_items[:12]
        ],
        "target_centroid_distance": _summarize_distances(centroid_distances),
        "target_centroid_margin": _summarize_margins(centroid_margins),
    }


def _centroid(items: list[list[float]]) -> list[float]:
    if not items:
        return []
    return [
        sum(hidden[dim] for hidden in items) / len(items)
        for dim in range(len(items[0]))
    ]


def _distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(
        sum(
            (left_value - right_value) ** 2
            for left_value, right_value in zip(left, right)
        )
    )


def _summarize_distances(distances: list[float]) -> dict[str, Any]:
    if not distances:
        return {"count": 0, "min": 0.0, "avg": 0.0, "max": 0.0}
    return {
        "count": len(distances),
        "min": min(distances),
        "avg": sum(distances) / len(distances),
        "max": max(distances),
    }


def _summarize_margins(margins: list[float]) -> dict[str, Any]:
    summary = _summarize_distances(margins)
    poorly_separated = sum(1 for margin in margins if margin <= 0.01)
    summary["poorly_separated"] = poorly_separated
    summary["poorly_separated_rate"] = (
        poorly_separated / len(margins) if margins else 0.0
    )
    return summary

