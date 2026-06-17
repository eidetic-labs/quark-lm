"""Evaluation for transformer-guided answer generators."""

from __future__ import annotations

from typing import Any

from tokenizer import CharTokenizer
from transformer_answer_generator_model import TransformerGuidedAnswerGenerator


def evaluate_answer_generator_records(
    generator: TransformerGuidedAnswerGenerator,
    model: Any,
    tokenizer: CharTokenizer,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    total_loss = 0.0
    for record in records:
        completion = generator.generate(model, tokenizer, record["prompt"])
        loss = generator.sequence_loss(model, tokenizer, record["prompt"], record["target"])
        total_loss += loss
        scored.append(
            {
                "id": record["id"],
                "target": record["target"],
                "completion": completion,
                "exact_match": completion == record["target"],
                "target_loss": loss,
                "completion_source": "transformer_guided_generator",
            }
        )
    exact = sum(1 for record in scored if record["exact_match"])
    failed = [record for record in scored if not record["exact_match"]]
    return {
        "count": len(scored),
        "exact": exact,
        "exact_rate": exact / len(scored) if scored else 0.0,
        "avg_target_loss": total_loss / len(scored) if scored else 0.0,
        "failed_records": failed,
    }
