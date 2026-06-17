"""Repair trainer adapters for direct-answer error modes."""

from __future__ import annotations

import argparse
import random
from typing import Any

from answer_model import AnswerExample
from autograd import Scalar
from tokenizer import CharTokenizer
from transformer_direct_answer_core import DirectAnswerLesson
from transformer_direct_answer_repair_objectives import (
    train_direct_answer_balanced_repair_unlikelihood,
    train_direct_answer_generated_prefix_recovery_unlikelihood,
    train_direct_answer_loop_escape_unlikelihood,
    train_direct_answer_sequence_repair_unlikelihood,
)


def train_balanced_repair_unlikelihood_step(
    args: argparse.Namespace,
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    lesson: DirectAnswerLesson,
    rng: random.Random,
    terminator: str,
    params: list[Scalar],
) -> float:
    return train_direct_answer_balanced_repair_unlikelihood(
        model,
        tokenizer,
        example,
        lesson,
        rng,
        args.direct_answer_learning_rate,
        args.direct_answer_negative_weight,
        args.direct_answer_positive_weight,
        terminator,
        params,
    )


def train_generated_prefix_recovery_unlikelihood_step(
    args: argparse.Namespace,
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    lesson: DirectAnswerLesson,
    rng: random.Random,
    terminator: str,
    params: list[Scalar],
) -> float:
    return train_direct_answer_generated_prefix_recovery_unlikelihood(
        model,
        tokenizer,
        example,
        lesson,
        rng,
        args.direct_answer_learning_rate,
        args.direct_answer_negative_weight,
        args.direct_answer_positive_weight,
        args.direct_answer_recovery_steps,
        terminator,
        params,
    )


def train_sequence_repair_unlikelihood_step(
    args: argparse.Namespace,
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    lesson: DirectAnswerLesson,
    rng: random.Random,
    terminator: str,
    params: list[Scalar],
) -> float:
    return train_direct_answer_sequence_repair_unlikelihood(
        model,
        tokenizer,
        example,
        lesson,
        rng,
        args.direct_answer_learning_rate,
        args.direct_answer_negative_weight,
        args.direct_answer_positive_weight,
        terminator,
        params,
    )


def train_loop_escape_unlikelihood_step(
    args: argparse.Namespace,
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    lesson: DirectAnswerLesson,
    rng: random.Random,
    terminator: str,
    params: list[Scalar],
) -> float:
    return train_direct_answer_loop_escape_unlikelihood(
        model,
        tokenizer,
        example,
        lesson,
        rng,
        args.direct_answer_learning_rate,
        args.direct_answer_negative_weight,
        args.direct_answer_positive_weight,
        terminator,
        params,
    )
