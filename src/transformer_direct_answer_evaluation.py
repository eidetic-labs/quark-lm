"""Direct-answer evaluation and context-coverage auditing helpers."""

from __future__ import annotations

from typing import Any

from answer_model import AnswerExample, semantic_feature_names
from tokenizer import CharTokenizer
from transformer_direct_answer_core import (
    direct_answer_sequence_nll,
)
from transformer_direct_answer_branch_context_evaluation import (
    audit_direct_answer_branch_context_coverage,
    summarize_branch_context_coverage_gate,
)
from transformer_direct_modes import ANSWER_TERMINATOR
from transformer_model import GenerationConfig


def evaluate_direct_answer_records(
    model: Any,
    tokenizer: CharTokenizer,
    records: list[dict[str, Any]],
    max_new_chars: int,
    terminator: str = ANSWER_TERMINATOR,
    generation_config: GenerationConfig | None = None,
) -> dict[str, Any]:
    generation_config = generation_config or GenerationConfig()
    scored: list[dict[str, Any]] = []
    total_loss = 0.0
    for record in records:
        generation = model.generate_with_trace(
            tokenizer,
            record["prompt"],
            max_new_chars,
            generation_config,
            stop_at=terminator if terminator else None,
        )
        completion = generation["text"]
        target = record["target"]
        example = AnswerExample(
            prompt=record["prompt"],
            target=target,
            source=f"eval:{record['id']}",
        )
        loss = direct_answer_sequence_nll(model, tokenizer, example, terminator)
        total_loss += loss
        scored.append(
            {
                "id": record["id"],
                "target": target,
                "completion": completion,
                "generation_trace": generation["trace"],
                "generation_cache": generation["cache"],
                "exact_match": completion == target,
                "target_loss": loss,
                "completion_source": "tiny_transformer_greedy_until_terminator"
                if terminator
                else "tiny_transformer_greedy_fixed_length",
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

def audit_prompt_context_coverage(
    records: list[dict[str, Any]],
    context_size: int,
    max_missing_records: int = 12,
) -> dict[str, Any]:
    audited = 0
    covered = 0
    missing_records: list[dict[str, Any]] = []
    for record in records:
        prompt = record["prompt"]
        full_features = set(semantic_feature_names(prompt.lower()))
        if not full_features:
            continue
        audited += 1
        context_text = prompt[-context_size:]
        context_features = set(semantic_feature_names(context_text.lower()))
        missing_features = sorted(full_features - context_features)
        if not missing_features:
            covered += 1
            continue
        if len(missing_records) < max_missing_records:
            missing_records.append(
                {
                    "id": record["id"],
                    "prompt_length": len(prompt),
                    "context_size": context_size,
                    "context_text": context_text,
                    "missing_features": missing_features,
                }
            )
    missing = audited - covered
    return {
        "semantic_records": audited,
        "covered": covered,
        "missing": missing,
        "covered_rate": covered / audited if audited else 1.0,
        "missing_records": missing_records,
    }
