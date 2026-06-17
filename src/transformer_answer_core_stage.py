"""Core answer-character training stage for transformer answer runs."""

from __future__ import annotations

import argparse
import random
from collections.abc import Callable
from typing import Any

from answer_model import AnswerExample
from tokenizer import CharTokenizer
from transformer_answer_training_steps import train_answer_char, train_answer_mixed_step
from transformer_training import LossAccumulator, ShuffledTrainingCursor

AnswerSnapshot = Callable[
    [int, float | None, float | None, float | None, float | None],
    dict[str, Any],
]


def train_core_answer_stage(
    args: argparse.Namespace,
    model: Any,
    tokenizer: CharTokenizer,
    training_pool: list[AnswerExample],
    rng: random.Random,
    training_candidates: list[str],
    snapshot: AnswerSnapshot,
) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline = snapshot(0, None, None, None, None)
    loss_accumulator = LossAccumulator()
    last_snapshot = baseline
    last_snapshot_step = 0
    training_cursor = ShuffledTrainingCursor(training_pool, rng)
    for step in range(1, args.steps + 1):
        example = training_cursor.next()
        if args.choice_loss_weight > 0.0 or args.target_loss_weight != 1.0:
            step_result = train_answer_mixed_step(
                model,
                tokenizer,
                example,
                rng,
                args.learning_rate,
                training_candidates,
                args.target_loss_weight,
                args.choice_loss_weight,
                args.choice_negatives,
                args.choice_max_chars,
            )
            loss_accumulator.add(
                step_result["loss"],
                step_result["target_loss"],
                step_result["choice_loss"],
                step_result["choice_candidate_count"],
            )
        else:
            loss = train_answer_char(model, tokenizer, example, rng, args.learning_rate)
            loss_accumulator.add(loss)
        if args.eval_every > 0 and step % args.eval_every == 0:
            averages = loss_accumulator.average(
                args.eval_every,
                include_choice=args.choice_loss_weight > 0.0,
            )
            last_snapshot = snapshot(
                step,
                averages["train_loss"],
                averages["train_target_loss"],
                averages["train_choice_loss"],
                averages["train_choice_candidates"],
            )
            last_snapshot_step = step
            print(f"step={step} train_loss={averages['train_loss']:.4f}")
            loss_accumulator.reset()

    if last_snapshot_step != args.steps:
        last_snapshot = snapshot(args.steps, None, None, None, None)
    return baseline, last_snapshot
