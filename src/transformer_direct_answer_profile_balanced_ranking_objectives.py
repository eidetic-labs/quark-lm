"""Profile-balanced branch ranking objective wrappers."""

from __future__ import annotations

import random
from typing import Any

from answer_model import AnswerExample
from autograd import Scalar
from tokenizer import CharTokenizer
from transformer_direct_answer_core import DirectAnswerLesson
from transformer_direct_answer_profile_balanced_batches import (
    direct_answer_profile_balanced_branch_batch,
    unprofiled_branch_records,
)
from transformer_direct_answer_repairs import train_direct_answer_lesson
from transformer_direct_modes import ANSWER_TERMINATOR
from transformer_profile_grouped_rank_collapse import (
    train_profile_grouped_rank_collapse,
)
from transformer_profile_balanced_target_depth import (
    PROFILE_BALANCED_DEFAULT_MIN_TARGETS_PER_PROFILE,
    PROFILE_BALANCED_RANK_COLLAPSE_MIN_TARGETS_PER_PROFILE,
)
from transformer_profile_balanced_target_floor_anchors import (
    profile_balanced_target_floor_anchors_from_examples,
)
from transformer_routing_repair_retention_anchors import (
    routing_repair_retention_anchor_batch,
)


def train_direct_answer_profile_balanced_branch_rank_margin_unlikelihood(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    margin_weight: float,
    branch_position: int,
    batch_size: int,
    hard_negative_count: int,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
    repair_target_profiles: list[str] | None = None,
) -> float:
    branches = _profile_balanced_branches(
        model,
        tokenizer,
        branch_examples,
        rng,
        branch_position,
        batch_size,
        terminator,
        repair_target_profiles=repair_target_profiles,
    )
    if not branches:
        return _fallback_loss(model, fallback_lesson, rng, learning_rate, params)
    return model.train_step_with_branch_rank_margin(
        unprofiled_branch_records(branches),
        learning_rate,
        negative_weight,
        positive_weight,
        margin_weight,
        hard_negative_count,
        params=params,
    )


def train_direct_answer_profile_balanced_branch_rank_collapse_unlikelihood(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    margin_weight: float,
    branch_position: int,
    batch_size: int,
    hard_negative_count: int,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
    repair_target_profiles: list[str] | None = None,
) -> float:
    branches = _profile_balanced_branches(
        model,
        tokenizer,
        branch_examples,
        rng,
        branch_position,
        batch_size,
        terminator,
        min_targets_per_profile=PROFILE_BALANCED_RANK_COLLAPSE_MIN_TARGETS_PER_PROFILE,
        repair_target_profiles=repair_target_profiles,
    )
    if not branches:
        return _fallback_loss(model, fallback_lesson, rng, learning_rate, params)
    return train_profile_grouped_rank_collapse(
        model,
        branches,
        learning_rate,
        negative_weight,
        positive_weight,
        margin_weight,
        negative_weight,
        hard_negative_count,
        params=params,
    )


def train_direct_answer_profile_balanced_branch_topk_softmax_unlikelihood(
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    candidate_weight: float,
    branch_position: int,
    batch_size: int,
    candidate_count: int,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
    repair_target_profiles: list[str] | None = None,
    eval_records: dict[str, list[dict[str, Any]]] | None = None,
) -> float:
    branches = _profile_balanced_branches(
        model,
        tokenizer,
        branch_examples,
        rng,
        branch_position,
        batch_size,
        terminator,
        repair_target_profiles=repair_target_profiles,
    )
    if not branches:
        return _fallback_loss(model, fallback_lesson, rng, learning_rate, params)
    retention_anchors = routing_repair_retention_anchor_batch(
        model,
        tokenizer,
        branch_examples,
        eval_records,
        rng,
        branch_position,
        batch_size,
        terminator,
    )
    target_floor_anchors = profile_balanced_target_floor_anchors_from_examples(
        model,
        tokenizer,
        branch_examples,
        rng,
        branch_position,
        batch_size,
        terminator,
        repair_target_profiles=repair_target_profiles,
    )
    return model.train_step_with_branch_retention_topk_softmax(
        unprofiled_branch_records(branches),
        retention_anchors,
        learning_rate,
        negative_weight,
        positive_weight,
        candidate_weight,
        candidate_count,
        params=params,
        target_floor_anchors=target_floor_anchors,
        representation_weight=candidate_weight,
    )


def _profile_balanced_branches(
    model: Any,
    tokenizer: CharTokenizer,
    branch_examples: list[AnswerExample],
    rng: random.Random,
    branch_position: int,
    batch_size: int,
    terminator: str,
    min_targets_per_profile: int = PROFILE_BALANCED_DEFAULT_MIN_TARGETS_PER_PROFILE,
    repair_target_profiles: list[str] | None = None,
) -> list[tuple[list[int], int, int, str]]:
    return direct_answer_profile_balanced_branch_batch(
        model,
        tokenizer,
        branch_examples,
        rng,
        branch_position,
        batch_size,
        terminator,
        min_targets_per_profile=min_targets_per_profile,
        repair_target_profiles=repair_target_profiles,
    )


def _fallback_loss(
    model: Any,
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    params: list[Scalar] | None,
) -> float:
    return train_direct_answer_lesson(
        model,
        fallback_lesson,
        rng,
        learning_rate,
        params=params,
    )
