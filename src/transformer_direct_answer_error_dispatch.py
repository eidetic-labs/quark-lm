"""Dispatch direct-answer error and repair objective modes."""

from __future__ import annotations

import argparse
import random
from typing import Any

from answer_model import AnswerExample
from autograd import Scalar
from tokenizer import CharTokenizer
from transformer_direct_answer_core import DirectAnswerLesson
from transformer_direct_answer_error_first_trainers import (
    train_first_error_step,
    train_first_error_unlikelihood_step,
)
from transformer_direct_answer_error_repair_trainers import (
    train_balanced_repair_unlikelihood_step,
    train_generated_prefix_recovery_unlikelihood_step,
    train_loop_escape_unlikelihood_step,
    train_sequence_repair_unlikelihood_step,
)
from transformer_direct_answer_error_rollout_trainers import (
    train_early_stop_unlikelihood_step,
    train_repeat_loop_unlikelihood_step,
    train_rollout_unlikelihood_step,
)


def train_direct_answer_error_repair_mode_step(
    *,
    args: argparse.Namespace,
    model: Any,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    lesson: DirectAnswerLesson,
    rng: random.Random,
    direct_step: int,
    terminator: str,
    params: list[Scalar],
) -> float | None:
    mode = args.direct_answer_mode

    def run(trainer: Any) -> float:
        return trainer(args, model, tokenizer, example, lesson, rng, terminator, params)

    if mode == "first-error":
        return run(train_first_error_step)
    if mode == "first-error-unlikelihood":
        return run(train_first_error_unlikelihood_step)
    if mode == "rollout-unlikelihood":
        return run(train_rollout_unlikelihood_step)
    if mode == "hybrid-unlikelihood":
        if direct_step % 2 == 0:
            return run(train_rollout_unlikelihood_step)
        return run(train_first_error_unlikelihood_step)
    if mode == "staged-unlikelihood":
        if direct_step <= args.direct_answer_steps // 2:
            return run(train_first_error_unlikelihood_step)
        return run(train_rollout_unlikelihood_step)
    if mode == "periodic-rollout-unlikelihood":
        if _on_interval(direct_step, args.direct_answer_rollout_interval):
            return run(train_rollout_unlikelihood_step)
        return run(train_first_error_unlikelihood_step)
    if mode == "early-stop-unlikelihood":
        return run(train_early_stop_unlikelihood_step)
    if mode == "periodic-early-stop-unlikelihood":
        if _on_interval(direct_step, args.direct_answer_rollout_interval):
            return run(train_early_stop_unlikelihood_step)
        return run(train_first_error_unlikelihood_step)
    if mode == "repeat-loop-unlikelihood":
        return run(train_repeat_loop_unlikelihood_step)
    if mode == "periodic-repeat-loop-unlikelihood":
        if _on_interval(direct_step, args.direct_answer_rollout_interval):
            return run(train_repeat_loop_unlikelihood_step)
        return run(train_first_error_unlikelihood_step)
    if mode == "balanced-repair-unlikelihood":
        return run(train_balanced_repair_unlikelihood_step)
    if mode == "periodic-balanced-repair-unlikelihood":
        if _on_interval(direct_step, args.direct_answer_rollout_interval):
            return run(train_balanced_repair_unlikelihood_step)
        return run(train_first_error_unlikelihood_step)
    if mode == "generated-prefix-recovery-unlikelihood":
        return run(train_generated_prefix_recovery_unlikelihood_step)
    if mode == "periodic-generated-prefix-recovery-unlikelihood":
        if _on_interval(direct_step, args.direct_answer_rollout_interval):
            return run(train_generated_prefix_recovery_unlikelihood_step)
        return run(train_first_error_unlikelihood_step)
    if mode == "sequence-repair-unlikelihood":
        return run(train_sequence_repair_unlikelihood_step)
    if mode == "periodic-sequence-repair-unlikelihood":
        if _on_interval(direct_step, args.direct_answer_rollout_interval):
            return run(train_sequence_repair_unlikelihood_step)
        return run(train_first_error_unlikelihood_step)
    if mode == "loop-escape-unlikelihood":
        return run(train_loop_escape_unlikelihood_step)
    if mode == "periodic-loop-escape-unlikelihood":
        if _on_interval(direct_step, args.direct_answer_rollout_interval):
            return run(train_loop_escape_unlikelihood_step)
        return run(train_first_error_unlikelihood_step)
    if mode == "periodic-sequence-loop-escape-unlikelihood":
        if _on_interval(direct_step, args.direct_answer_rollout_interval):
            return run(train_loop_escape_unlikelihood_step)
        if _on_interval(direct_step, args.direct_answer_sequence_interval):
            return run(train_sequence_repair_unlikelihood_step)
        return run(train_first_error_unlikelihood_step)
    return None


def _on_interval(direct_step: int, interval: int) -> bool:
    return direct_step % max(1, interval) == 0
