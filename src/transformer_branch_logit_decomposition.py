"""Dominant-token decomposition for branch logit-prior diagnostics."""

from __future__ import annotations

from typing import Any

from tokenizer import CharTokenizer


def hidden_logit_contribution(
    model: Any,
    hidden: list[float],
    token_id: int,
) -> float:
    return sum(
        hidden[dim] * model.wout[dim][token_id].data
        for dim in range(len(hidden))
    )


def summarize_logit_decomposition(items: list[dict[str, float]]) -> dict[str, Any]:
    if not items:
        return {
            "count": 0,
            "avg_bias_advantage": 0.0,
            "avg_hidden_advantage": 0.0,
            "avg_logit_advantage": 0.0,
            "bias_share_of_positive_advantage": 0.0,
            "dominant_logit_win_rate": 0.0,
            "primary_pressure": "none",
        }
    avg_bias = sum(item["bias_advantage"] for item in items) / len(items)
    avg_hidden = sum(item["hidden_advantage"] for item in items) / len(items)
    avg_logit = sum(item["logit_advantage"] for item in items) / len(items)
    positive_bias = max(avg_bias, 0.0)
    positive_hidden = max(avg_hidden, 0.0)
    positive_total = positive_bias + positive_hidden
    bias_share = positive_bias / positive_total if positive_total else 0.0
    if avg_logit <= 0.0:
        primary_pressure = "target_not_losing_to_dominant"
    elif bias_share >= 0.67:
        primary_pressure = "output_bias"
    elif bias_share <= 0.33:
        primary_pressure = "hidden_projection"
    else:
        primary_pressure = "mixed_bias_hidden"
    return {
        "count": len(items),
        "avg_bias_advantage": avg_bias,
        "avg_hidden_advantage": avg_hidden,
        "avg_logit_advantage": avg_logit,
        "bias_share_of_positive_advantage": bias_share,
        "dominant_logit_win_rate": (
            sum(1 for item in items if item["logit_advantage"] > 0.0)
            / len(items)
        ),
        "primary_pressure": primary_pressure,
    }


def dominant_vs_target_decomposition(
    *,
    model: Any,
    tokenizer: CharTokenizer,
    rows: list[dict[str, Any]],
    dominant_id: int | None,
    max_sample_records: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    decompositions: list[dict[str, float]] = []
    failed_decompositions: list[dict[str, float]] = []
    sample_records: list[dict[str, Any]] = []
    if dominant_id is None:
        return {
            "all_records": summarize_logit_decomposition(decompositions),
            "failed_records": summarize_logit_decomposition(failed_decompositions),
        }, sample_records

    dominant_bias = model.bout[dominant_id].data
    for row in rows:
        target_id = int(row["target_id"])
        hidden = row["hidden"]
        target_bias = model.bout[target_id].data
        dominant_hidden = hidden_logit_contribution(model, hidden, dominant_id)
        target_hidden = hidden_logit_contribution(model, hidden, target_id)
        dominant_logit = row["logits"][dominant_id]
        target_logit = row["logits"][target_id]
        decomposition = {
            "bias_advantage": dominant_bias - target_bias,
            "hidden_advantage": dominant_hidden - target_hidden,
            "logit_advantage": dominant_logit - target_logit,
        }
        decompositions.append(decomposition)
        if int(row["predicted_id"]) != target_id:
            failed_decompositions.append(decomposition)
            if len(sample_records) < max_sample_records:
                sample_records.append(
                    {
                        "id": row["id"],
                        "target_token": tokenizer.itos[target_id],
                        "predicted_token": tokenizer.itos[int(row["predicted_id"])],
                        "dominant_token": tokenizer.itos[dominant_id],
                        "target_rank": row["target_rank"],
                        "bias_advantage": decomposition["bias_advantage"],
                        "hidden_advantage": decomposition["hidden_advantage"],
                        "logit_advantage": decomposition["logit_advantage"],
                    }
                )

    return {
        "all_records": summarize_logit_decomposition(decompositions),
        "failed_records": summarize_logit_decomposition(failed_decompositions),
    }, sample_records
