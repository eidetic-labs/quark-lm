"""Branch token and representation diagnostics for direct-answer runs."""

from __future__ import annotations

from collections import Counter
from typing import Any

from answer_model import AnswerExample
from tokenizer import CharTokenizer
from transformer_branch_representation_profiles import (
    direct_answer_branch_representation_profile,
)
from transformer_direct_answer_core import direct_answer_branch_context
from transformer_direct_modes import ANSWER_TERMINATOR


def direct_answer_branch_profile(
    model: Any,
    tokenizer: CharTokenizer,
    records: list[dict[str, Any]],
    branch_position: int,
    terminator: str = ANSWER_TERMINATOR,
    max_failed_records: int = 12,
) -> dict[str, Any]:
    predicted_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    confusion_counts: Counter[str] = Counter()
    failed_records: list[dict[str, Any]] = []
    total_target_prob = 0.0
    total_predicted_prob = 0.0
    total_target_margin = 0.0
    total_target_rank = 0.0
    target_top3 = 0
    target_top5 = 0
    profiled = 0
    correct = 0
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
        context, target_id, position = branch
        probs = model.predict(context)
        ranked_ids = sorted(
            range(len(probs)),
            key=lambda index: (-probs[index], tokenizer.itos[index], index),
        )
        predicted_id = ranked_ids[0]
        target_rank = ranked_ids.index(target_id) + 1
        target_prob = probs[target_id]
        predicted_prob = probs[predicted_id]
        strongest_non_target = max(
            (prob for index, prob in enumerate(probs) if index != target_id),
            default=0.0,
        )
        target_margin = target_prob - strongest_non_target
        target_token = tokenizer.itos[target_id]
        predicted_token = tokenizer.itos[predicted_id]

        profiled += 1
        total_target_prob += target_prob
        total_predicted_prob += predicted_prob
        total_target_margin += target_margin
        total_target_rank += target_rank
        if target_rank <= 3:
            target_top3 += 1
        if target_rank <= 5:
            target_top5 += 1
        target_counts[target_token] += 1
        predicted_counts[predicted_token] += 1
        confusion_counts[f"{target_token!r}->{predicted_token!r}"] += 1
        if predicted_id == target_id:
            correct += 1
        elif len(failed_records) < max_failed_records:
            failed_records.append(
                {
                    "id": record["id"],
                    "target": record["target"],
                    "branch_position": position,
                    "target_token": target_token,
                    "predicted_token": predicted_token,
                    "target_prob": target_prob,
                    "predicted_prob": predicted_prob,
                    "target_margin": target_margin,
                    "target_rank": target_rank,
                    "top_predictions": [
                        {
                            "token": tokenizer.itos[index],
                            "prob": probs[index],
                        }
                        for index in ranked_ids[:5]
                    ],
                }
            )

    def top_items(counter: Counter[str]) -> list[dict[str, Any]]:
        return [
            {"value": value, "count": count}
            for value, count in counter.most_common(12)
        ]

    target_token_values = set(target_counts)
    predicted_token_values = set(predicted_counts)
    covered_target_tokens = target_token_values & predicted_token_values
    dominant_predicted_token = None
    dominant_predicted_count = 0
    if predicted_counts:
        dominant_predicted_token, dominant_predicted_count = (
            predicted_counts.most_common(1)[0]
        )
    missing_target_tokens = [
        {"value": value, "count": count}
        for value, count in target_counts.most_common()
        if value not in predicted_token_values
    ]
    target_unique = len(target_token_values)
    predicted_unique = len(predicted_token_values)

    return {
        "branch_position": branch_position,
        "count": profiled,
        "skipped": skipped,
        "correct": correct,
        "accuracy": correct / profiled if profiled else 0.0,
        "avg_target_prob": total_target_prob / profiled if profiled else 0.0,
        "avg_predicted_prob": total_predicted_prob / profiled if profiled else 0.0,
        "avg_target_margin": total_target_margin / profiled if profiled else 0.0,
        "target_rank": {
            "avg": total_target_rank / profiled if profiled else 0.0,
            "top1_rate": correct / profiled if profiled else 0.0,
            "top3_rate": target_top3 / profiled if profiled else 0.0,
            "top5_rate": target_top5 / profiled if profiled else 0.0,
            "vocab_size": model.config.vocab_size,
        },
        "target_tokens": top_items(target_counts),
        "predicted_tokens": top_items(predicted_counts),
        "confusions": top_items(confusion_counts),
        "diversity": {
            "target_unique": target_unique,
            "predicted_unique": predicted_unique,
            "target_token_coverage": (
                len(covered_target_tokens) / target_unique
                if target_unique
                else 0.0
            ),
            "dominant_predicted_token": dominant_predicted_token,
            "dominant_predicted_count": dominant_predicted_count,
            "dominant_predicted_rate": (
                dominant_predicted_count / profiled
                if profiled
                else 0.0
            ),
            "collapsed": profiled > 1 and target_unique > 1 and predicted_unique == 1,
            "missing_target_tokens": missing_target_tokens,
        },
        "failed_records": failed_records,
    }
