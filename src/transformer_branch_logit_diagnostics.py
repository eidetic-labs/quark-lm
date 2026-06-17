"""Branch logit-prior diagnostics for direct-answer runs."""

from __future__ import annotations

from typing import Any

from tokenizer import CharTokenizer
from transformer_branch_logit_bias import (
    branch_logit_bias_rankings,
    branch_logit_bias_summary,
    branch_logit_top_bias_tokens,
    branch_logit_top_token_items,
    missing_branch_target_ids,
)
from transformer_branch_logit_decomposition import dominant_vs_target_decomposition
from transformer_branch_logit_rows import collect_branch_logit_prior_rows
from transformer_branch_diversity_summary import summarize_branch_diversity_target
from transformer_direct_modes import ANSWER_TERMINATOR


def direct_answer_branch_logit_prior_profile(
    model: Any,
    tokenizer: CharTokenizer,
    records: list[dict[str, Any]],
    branch_position: int,
    terminator: str = ANSWER_TERMINATOR,
    max_sample_records: int = 8,
) -> dict[str, Any]:
    collected = collect_branch_logit_prior_rows(
        model,
        tokenizer,
        records,
        branch_position,
        terminator,
    )
    if collected.predicted_counts:
        dominant_id, dominant_count = collected.predicted_counts.most_common(1)[0]
    else:
        dominant_id = None
        dominant_count = 0
    bias_rankings = branch_logit_bias_rankings(model, tokenizer)
    missing_target_token_ids = missing_branch_target_ids(
        collected.target_counts,
        collected.predicted_counts,
        tokenizer,
    )
    missing_target_biases = [
        model.bout[token_id].data for token_id in missing_target_token_ids
    ]
    dominant_bias_value = (
        model.bout[dominant_id].data if dominant_id is not None else 0.0
    )
    decomposition, sample_records = dominant_vs_target_decomposition(
        model=model,
        tokenizer=tokenizer,
        rows=collected.rows,
        dominant_id=dominant_id,
        max_sample_records=max_sample_records,
    )

    return {
        "branch_position": branch_position,
        "count": len(collected.rows),
        "skipped": collected.skipped,
        "target_tokens": branch_logit_top_token_items(
            collected.target_counts,
            tokenizer,
        ),
        "predicted_tokens": branch_logit_top_token_items(
            collected.predicted_counts,
            tokenizer,
        ),
        "dominant_predicted_token": (
            tokenizer.itos[dominant_id] if dominant_id is not None else None
        ),
        "dominant_predicted_count": dominant_count,
        "dominant_predicted_rate": (
            dominant_count / len(collected.rows) if collected.rows else 0.0
        ),
        "dominant_token_bias": dominant_bias_value,
        "dominant_token_bias_rank": (
            bias_rankings.get(dominant_id) if dominant_id is not None else None
        ),
        "missing_target_tokens": [
            {
                "value": tokenizer.itos[token_id],
                "count": collected.target_counts[token_id],
            }
            for token_id in missing_target_token_ids[:12]
        ],
        "missing_target_bias": branch_logit_bias_summary(missing_target_biases),
        "dominant_vs_missing_bias_advantage": (
            dominant_bias_value
            - (
                sum(missing_target_biases) / len(missing_target_biases)
                if missing_target_biases
                else 0.0
            )
        ),
        "top_bias_tokens": branch_logit_top_bias_tokens(
            model,
            tokenizer,
            bias_rankings,
        ),
        "dominant_vs_target_decomposition": decomposition,
        "sample_records": sample_records,
    }
