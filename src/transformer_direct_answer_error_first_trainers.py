"""First-error trainer adapters for direct-answer repair modes."""

from __future__ import annotations

import argparse
import random
from typing import Any

from answer_model import AnswerExample
from autograd import Scalar
from tokenizer import CharTokenizer
from transformer_direct_answer_core import DirectAnswerLesson
from transformer_direct_answer_repairs import (
    train_direct_answer_first_error,
    train_direct_answer_first_error_unlikelihood,
)


def train_first_error_step(
    args: argparse.Namespace,
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    lesson: DirectAnswerLesson,
    rng: random.Random,
    terminator: str,
    params: list[Scalar],
) -> float:
    return train_direct_answer_first_error(
        model,
        tokenizer,
        example,
        lesson,
        rng,
        args.direct_answer_learning_rate,
        terminator,
        params,
    )


def train_first_error_unlikelihood_step(
    args: argparse.Namespace,
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    lesson: DirectAnswerLesson,
    rng: random.Random,
    terminator: str,
    params: list[Scalar],
) -> float:
    return train_direct_answer_first_error_unlikelihood(
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
