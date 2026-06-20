"""Profile-balanced branch objective adapters."""

from __future__ import annotations

from typing import Any

from transformer_direct_answer_branch_ranking_objectives import (
    train_direct_answer_profile_balanced_branch_rank_margin_unlikelihood,
    train_direct_answer_profile_balanced_branch_topk_softmax_unlikelihood,
)
from transformer_direct_answer_branch_retention_objectives import (
    train_direct_answer_profile_balanced_retention_rank_margin_unlikelihood,
)


def train_profile_balanced_branch_rank_margin(
    step: Any,
) -> float:
    args = step.args
    return train_direct_answer_profile_balanced_branch_rank_margin_unlikelihood(
        step.model,
        step.tokenizer,
        step.example,
        step.branch_examples,
        step.lesson,
        step.rng,
        args.direct_answer_learning_rate,
        args.direct_answer_negative_weight,
        args.direct_answer_positive_weight,
        args.direct_answer_contrast_weight,
        args.direct_answer_branch_position,
        args.direct_answer_branch_batch_size,
        args.direct_answer_hard_negatives,
        step.terminator,
        step.params,
    )


def train_profile_balanced_branch_topk_softmax(
    step: Any,
) -> float:
    args = step.args
    return train_direct_answer_profile_balanced_branch_topk_softmax_unlikelihood(
        step.model,
        step.tokenizer,
        step.example,
        step.branch_examples,
        step.lesson,
        step.rng,
        args.direct_answer_learning_rate,
        args.direct_answer_negative_weight,
        args.direct_answer_positive_weight,
        args.direct_answer_contrast_weight,
        args.direct_answer_branch_position,
        args.direct_answer_branch_batch_size,
        args.direct_answer_hard_negatives,
        step.terminator,
        step.params,
    )


def train_profile_balanced_retention_branch_rank_margin(
    step: Any,
) -> float:
    args = step.args
    return train_direct_answer_profile_balanced_retention_rank_margin_unlikelihood(
        step.model,
        step.tokenizer,
        step.example,
        step.branch_examples,
        step.lesson,
        step.rng,
        args.direct_answer_learning_rate,
        args.direct_answer_negative_weight,
        args.direct_answer_positive_weight,
        args.direct_answer_contrast_weight,
        args.direct_answer_branch_position,
        args.direct_answer_branch_batch_size,
        args.direct_answer_hard_negatives,
        step.terminator,
        step.params,
    )
