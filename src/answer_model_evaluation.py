"""Evaluation summaries for answer model checkpoints."""

from __future__ import annotations

from typing import Any

from answer_model_softmax import AnswerSoftmax


def evaluate_records(
    model: AnswerSoftmax,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    scored = []
    total_loss = 0.0
    for record in records:
        prediction = model.predict(record["prompt"])
        loss = model.loss(record["prompt"], record["target"])
        total_loss += loss
        scored.append(
            {
                "id": record["id"],
                "target": record["target"],
                "prediction": prediction,
                "exact_match": prediction == record["target"],
                "target_loss": loss,
            }
        )
    exact = sum(1 for record in scored if record["exact_match"])
    return {
        "count": len(scored),
        "exact": exact,
        "exact_rate": exact / len(scored) if scored else 0.0,
        "avg_target_loss": total_loss / len(scored) if scored else 0.0,
        "records": scored,
    }


def summarize_eval(
    model: AnswerSoftmax,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    result = evaluate_records(model, records)
    failed_records = [record for record in result["records"] if not record["exact_match"]]
    return {
        "count": result["count"],
        "exact": result["exact"],
        "exact_rate": result["exact_rate"],
        "avg_target_loss": result["avg_target_loss"],
        "failed_records": failed_records,
    }
