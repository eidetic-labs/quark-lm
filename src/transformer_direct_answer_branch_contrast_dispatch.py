"""Dispatch branch rank, contrast, and span-repair direct-answer modes."""

from __future__ import annotations

import argparse
import random
from typing import Any

from answer_model import AnswerExample
from autograd import Scalar
from tokenizer import CharTokenizer
from transformer_direct_answer_branch_contrast_adapters import (
    BranchContrastModeStep,
    train_branch_contrast,
    train_branch_rank_margin,
    train_branch_repair,
    train_branch_representation_contrast,
    train_branch_span_contrast,
    train_branch_span_repair,
    train_branch_topk_softmax,
    train_first_error_unlikelihood,
    train_hard_branch_contrast,
)
from transformer_direct_answer_core import DirectAnswerLesson
from transformer_direct_answer_branch_profile_balanced_adapters import (
    train_profile_balanced_branch_rank_margin,
    train_profile_balanced_branch_topk_softmax,
)


BRANCH_CONTRAST_DIRECT_ANSWER_MODES = frozenset(
    {
        "branch-rank-margin-unlikelihood",
        "branch-balanced-rank-margin-unlikelihood",
        "branch-profile-balanced-rank-margin-unlikelihood",
        "branch-topk-softmax-unlikelihood",
        "branch-balanced-topk-softmax-unlikelihood",
        "branch-profile-balanced-topk-softmax-unlikelihood",
        "periodic-branch-representation-contrast-unlikelihood",
        "branch-span-repair-unlikelihood",
        "periodic-branch-span-repair-unlikelihood",
        "branch-contrast-unlikelihood",
        "periodic-branch-contrast-unlikelihood",
        "branch-span-contrast-unlikelihood",
        "periodic-branch-span-contrast-unlikelihood",
        "hard-branch-contrast-unlikelihood",
        "periodic-hard-branch-contrast-unlikelihood",
        "periodic-branch-repair-contrast-unlikelihood",
        "periodic-branch-span-repair-contrast-unlikelihood",
        "periodic-hard-branch-repair-contrast-unlikelihood",
    }
)


def train_direct_answer_branch_contrast_mode_step(
    *,
    args: argparse.Namespace,
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    lesson: DirectAnswerLesson,
    branch_examples: list[AnswerExample],
    rng: random.Random,
    direct_step: int,
    terminator: str,
    params: list[Scalar],
) -> float:
    step = BranchContrastModeStep(
        args=args,
        model=model,
        tokenizer=tokenizer,
        example=example,
        lesson=lesson,
        branch_examples=branch_examples,
        rng=rng,
        terminator=terminator,
        params=params,
    )
    mode = args.direct_answer_mode
    if mode == "branch-profile-balanced-rank-margin-unlikelihood":
        return train_profile_balanced_branch_rank_margin(step)
    if mode == "branch-profile-balanced-topk-softmax-unlikelihood":
        return train_profile_balanced_branch_topk_softmax(step)
    if mode in _RANK_MARGIN_MODES:
        return train_branch_rank_margin(
            step,
            balance_targets=mode.startswith("branch-balanced-"),
        )
    if mode in _TOPK_SOFTMAX_MODES:
        return train_branch_topk_softmax(
            step,
            balance_targets=mode.startswith("branch-balanced-"),
        )
    if mode == "periodic-branch-representation-contrast-unlikelihood":
        if _on_interval(direct_step, args.direct_answer_rollout_interval):
            return train_branch_representation_contrast(step)
        return train_branch_repair(step)
    if mode == "branch-span-repair-unlikelihood":
        return train_branch_span_repair(step)
    if mode == "periodic-branch-span-repair-unlikelihood":
        if _on_interval(direct_step, args.direct_answer_rollout_interval):
            return train_branch_span_repair(step)
        return train_first_error_unlikelihood(step)
    if mode == "branch-contrast-unlikelihood":
        return train_branch_contrast(step)
    if mode == "periodic-branch-contrast-unlikelihood":
        if _on_interval(direct_step, args.direct_answer_rollout_interval):
            return train_branch_contrast(step)
        return train_first_error_unlikelihood(step)
    if mode == "branch-span-contrast-unlikelihood":
        return train_branch_span_contrast(step)
    if mode == "periodic-branch-span-contrast-unlikelihood":
        if _on_interval(direct_step, args.direct_answer_rollout_interval):
            return train_branch_span_contrast(step)
        return train_first_error_unlikelihood(step)
    if mode == "hard-branch-contrast-unlikelihood":
        return train_hard_branch_contrast(step)
    if mode == "periodic-hard-branch-contrast-unlikelihood":
        if _on_interval(direct_step, args.direct_answer_rollout_interval):
            return train_hard_branch_contrast(step)
        return train_first_error_unlikelihood(step)
    if mode == "periodic-branch-repair-contrast-unlikelihood":
        if _on_interval(direct_step, args.direct_answer_rollout_interval):
            return train_branch_contrast(step)
        return train_branch_repair(step)
    if mode == "periodic-branch-span-repair-contrast-unlikelihood":
        if _on_interval(direct_step, args.direct_answer_rollout_interval):
            return train_branch_span_contrast(step)
        return train_branch_span_repair(step)
    if mode == "periodic-hard-branch-repair-contrast-unlikelihood":
        if _on_interval(direct_step, args.direct_answer_rollout_interval):
            return train_hard_branch_contrast(step)
        return train_branch_repair(step)
    raise ValueError(f"Unsupported branch contrast direct-answer mode: {mode}")


def _on_interval(direct_step: int, interval: int) -> bool:
    return direct_step % max(1, interval) == 0


_RANK_MARGIN_MODES = frozenset(
    {
        "branch-rank-margin-unlikelihood",
        "branch-balanced-rank-margin-unlikelihood",
    }
)
_TOPK_SOFTMAX_MODES = frozenset(
    {
        "branch-topk-softmax-unlikelihood",
        "branch-balanced-topk-softmax-unlikelihood",
    }
)
