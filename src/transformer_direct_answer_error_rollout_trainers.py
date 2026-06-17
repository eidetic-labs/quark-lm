"""Rollout-style trainer adapters for direct-answer error modes."""

from __future__ import annotations

import argparse
import random
from typing import Any

from answer_model import AnswerExample
from autograd import Scalar
from tokenizer import CharTokenizer
from transformer_direct_answer_core import DirectAnswerLesson
from transformer_direct_answer_repair_objectives import (
    train_direct_answer_early_stop_unlikelihood,
    train_direct_answer_repeat_loop_unlikelihood,
    train_direct_answer_rollout_unlikelihood,
)


def train_rollout_unlikelihood_step(
    args: argparse.Namespace,
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    lesson: DirectAnswerLesson,
    rng: random.Random,
    terminator: str,
    params: list[Scalar],
) -> float:
    return train_direct_answer_rollout_unlikelihood(
        model,
        tokenizer,
        example,
        lesson,
        rng,
        args.direct_answer_learning_rate,
        args.direct_answer_negative_weight,
        terminator,
        params,
    )


def train_early_stop_unlikelihood_step(
    args: argparse.Namespace,
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    lesson: DirectAnswerLesson,
    rng: random.Random,
    terminator: str,
    params: list[Scalar],
) -> float:
    return train_direct_answer_early_stop_unlikelihood(
        model,
        tokenizer,
        example,
        lesson,
        rng,
        args.direct_answer_learning_rate,
        args.direct_answer_negative_weight,
        terminator,
        params,
    )


def train_repeat_loop_unlikelihood_step(
    args: argparse.Namespace,
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    lesson: DirectAnswerLesson,
    rng: random.Random,
    terminator: str,
    params: list[Scalar],
) -> float:
    return train_direct_answer_repeat_loop_unlikelihood(
        model,
        tokenizer,
        example,
        lesson,
        rng,
        args.direct_answer_learning_rate,
        args.direct_answer_negative_weight,
        terminator,
        params,
    )
