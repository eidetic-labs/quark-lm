"""Shared payload for one basic branch direct-answer objective step."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Any

from answer_model import AnswerExample
from autograd import Scalar
from tokenizer import CharTokenizer
from transformer_direct_answer_core import DirectAnswerLesson


@dataclass(frozen=True)
class BranchBasicModeStep:
    args: argparse.Namespace
    model: Any
    tokenizer: CharTokenizer
    example: AnswerExample
    lesson: DirectAnswerLesson
    branch_examples: list[AnswerExample]
    rng: random.Random
    terminator: str
    params: list[Scalar]
