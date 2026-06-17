"""Dispatch basic branch objective modes for direct-answer training."""

from __future__ import annotations

import argparse
import random
from collections.abc import Callable
from typing import Any

from answer_model import AnswerExample
from autograd import Scalar
from tokenizer import CharTokenizer
from transformer_direct_answer_branch_basic_modes import BASIC_BRANCH_DIRECT_ANSWER_MODES
from transformer_direct_answer_branch_basic_step import BranchBasicModeStep
from transformer_direct_answer_branch_basic_trainers import (
    train_branch_batch_contrast,
    train_branch_collapse,
    train_branch_diversity,
    train_branch_hidden_projection_margin,
    train_branch_repair,
    train_branch_target_margin,
    train_branch_target_softmax,
    train_first_error_unlikelihood,
)
from transformer_direct_answer_core import DirectAnswerLesson


BasicBranchTrainer = Callable[[BranchBasicModeStep], float]


_BASIC_BRANCH_TRAINERS: dict[str, BasicBranchTrainer] = {
    "branch-repair-unlikelihood": train_branch_repair,
    "branch-collapse-unlikelihood": train_branch_collapse,
    "branch-batch-contrast-unlikelihood": train_branch_batch_contrast,
    "branch-diversity-unlikelihood": train_branch_diversity,
    "branch-target-softmax-unlikelihood": train_branch_target_softmax,
    "branch-target-margin-unlikelihood": train_branch_target_margin,
    "branch-hidden-projection-margin-unlikelihood": train_branch_hidden_projection_margin,
}


_PERIODIC_BRANCH_TRAINERS: dict[
    str,
    tuple[BasicBranchTrainer, BasicBranchTrainer],
] = {
    "periodic-branch-repair-unlikelihood": (
        train_branch_repair,
        train_first_error_unlikelihood,
    ),
    "periodic-branch-collapse-unlikelihood": (
        train_branch_collapse,
        train_branch_repair,
    ),
    "periodic-branch-batch-contrast-unlikelihood": (
        train_branch_batch_contrast,
        train_branch_repair,
    ),
    "periodic-branch-diversity-unlikelihood": (
        train_branch_diversity,
        train_branch_repair,
    ),
    "periodic-branch-target-softmax-unlikelihood": (
        train_branch_target_softmax,
        train_branch_repair,
    ),
    "periodic-branch-target-margin-unlikelihood": (
        train_branch_target_margin,
        train_branch_repair,
    ),
}


def train_direct_answer_branch_basic_mode_step(
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
    step = BranchBasicModeStep(
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
    if mode in _BASIC_BRANCH_TRAINERS:
        return _BASIC_BRANCH_TRAINERS[mode](step)
    if mode in _PERIODIC_BRANCH_TRAINERS:
        primary_trainer, fallback_trainer = _PERIODIC_BRANCH_TRAINERS[mode]
        if _on_interval(direct_step, args.direct_answer_rollout_interval):
            return primary_trainer(step)
        return fallback_trainer(step)
    raise ValueError(f"Unsupported basic branch direct-answer mode: {mode}")


def _on_interval(direct_step: int, interval: int) -> bool:
    return direct_step % max(1, interval) == 0


__all__ = [
    "BASIC_BRANCH_DIRECT_ANSWER_MODES",
    "train_direct_answer_branch_basic_mode_step",
]
