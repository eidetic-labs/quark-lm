"""Auxiliary selector and generator stages for transformer answer training."""

from __future__ import annotations

import argparse
import random
from typing import Any

from answer_model import AnswerExample
from tokenizer import CharTokenizer
from transformer_answer_training_snapshots import (
    generator_snapshot_record,
    selector_snapshot_record,
)
from transformer_answer_training_steps import sampled_choice_candidates
from transformer_answer_generator import (
    build_transformer_answer_generator,
    train_transformer_answer_generator_lesson,
    transformer_answer_generator_lesson,
    transformer_answer_generator_training_pool,
)
from transformer_answer_selector import build_answer_selector
from transformer_experiment import TRAINING_DATA_DESCRIPTION
from transformer_training import JsonlHistoryWriter, ShuffledTrainingCursor


def train_answer_selector_stage(
    args: argparse.Namespace,
    model: Any,
    tokenizer: CharTokenizer,
    examples: list[AnswerExample],
    training_pool: list[AnswerExample],
    eval_records: dict[str, list[dict[str, Any]]],
    eval_candidates: dict[str, list[str]],
    candidates: list[str],
) -> dict[str, Any] | None:
    if args.selector_steps <= 0:
        return None
    selector = build_answer_selector(examples, args.seed + 101)
    selector_rng = random.Random(args.seed + 101)
    selector_history_path = args.run / "answer_selector_metrics.jsonl"
    selector_history_writer = JsonlHistoryWriter(selector_history_path)

    def selector_snapshot(step: int, train_loss: float | None) -> dict[str, Any]:
        return selector_history_writer.append(
            selector_snapshot_record(
                model,
                tokenizer,
                selector,
                eval_records,
                eval_candidates,
                candidates,
                args.candidate_scope,
                args.max_new_chars,
                args.selector_emit_completions,
                step,
                train_loss,
            )
        )

    selector_baseline = selector_snapshot(0, None)
    running_selector_loss = 0.0
    last_selector_snapshot = selector_baseline
    last_selector_snapshot_step = 0
    selector_training_cursor = ShuffledTrainingCursor(training_pool, selector_rng)
    selector_candidates = selector.config.labels
    for selector_step in range(1, args.selector_steps + 1):
        example = selector_training_cursor.next()
        if args.selector_negatives > 0:
            selector_batch = sampled_choice_candidates(
                example.target,
                selector_candidates,
                selector_rng,
                args.selector_negatives,
            )
        else:
            selector_batch = selector_candidates
        running_selector_loss += selector.train_step(
            example,
            args.selector_learning_rate,
            selector_batch,
        )
        if args.selector_eval_every > 0 and selector_step % args.selector_eval_every == 0:
            train_loss = running_selector_loss / args.selector_eval_every
            last_selector_snapshot = selector_snapshot(selector_step, train_loss)
            last_selector_snapshot_step = selector_step
            print(f"selector_step={selector_step} train_loss={train_loss:.4f}")
            running_selector_loss = 0.0

    if last_selector_snapshot_step != args.selector_steps:
        last_selector_snapshot = selector_snapshot(args.selector_steps, None)

    selector_checkpoint_path = args.run / "answer_selector.json"
    selector.save(selector_checkpoint_path)
    return {
        "architecture": "closed-world-answer-candidate-selector",
        "checkpoint": str(selector_checkpoint_path),
        "history": str(selector_history_path),
        "steps": args.selector_steps,
        "learning_rate": args.selector_learning_rate,
        "selector_negatives": args.selector_negatives,
        "selector_eval_every": args.selector_eval_every,
        "selector_emit_completions": args.selector_emit_completions,
        "labels": len(selector.config.labels),
        "features": len(selector.config.features),
        "candidate_scope": args.candidate_scope,
        "baseline": selector_baseline,
        "final": last_selector_snapshot,
        "pretrained_weights": False,
        "pretrained_tokenizer": False,
        "external_embeddings": False,
        "training_data": TRAINING_DATA_DESCRIPTION,
    }


def train_answer_generator_stage(
    args: argparse.Namespace,
    model: Any,
    tokenizer: CharTokenizer,
    examples: list[AnswerExample],
    eval_records: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    if args.generator_steps <= 0:
        return None
    generator_training_pool = transformer_answer_generator_training_pool(examples)
    generator = build_transformer_answer_generator(
        examples,
        model,
        tokenizer,
        args.seed + 211,
        args.generator_max_answer_chars,
        args.generator_transformer_top_k,
    )
    generator_rng = random.Random(args.seed + 211)
    generator_history_path = args.run / "answer_generator_metrics.jsonl"
    generator_history_writer = JsonlHistoryWriter(generator_history_path)

    def generator_snapshot(step: int, train_loss: float | None) -> dict[str, Any]:
        return generator_history_writer.append(
            generator_snapshot_record(
                generator,
                model,
                tokenizer,
                eval_records,
                step,
                train_loss,
            )
        )

    generator_baseline = generator_snapshot(0, None)
    generator_lessons = {
        example: transformer_answer_generator_lesson(
            generator,
            model,
            tokenizer,
            example,
        )
        for example in sorted(
            set(generator_training_pool),
            key=lambda item: (item.prompt, item.target, item.source),
        )
    }
    running_generator_loss = 0.0
    last_generator_snapshot = generator_baseline
    last_generator_snapshot_step = 0
    generator_training_cursor = ShuffledTrainingCursor(
        generator_training_pool,
        generator_rng,
    )
    for generator_step in range(1, args.generator_steps + 1):
        example = generator_training_cursor.next()
        running_generator_loss += train_transformer_answer_generator_lesson(
            generator,
            generator_lessons[example],
            args.generator_learning_rate,
        )
        if args.generator_eval_every > 0 and generator_step % args.generator_eval_every == 0:
            train_loss = running_generator_loss / args.generator_eval_every
            last_generator_snapshot = generator_snapshot(generator_step, train_loss)
            last_generator_snapshot_step = generator_step
            print(f"generator_step={generator_step} train_loss={train_loss:.4f}")
            running_generator_loss = 0.0

    if last_generator_snapshot_step != args.generator_steps:
        last_generator_snapshot = generator_snapshot(args.generator_steps, None)

    generator_checkpoint_path = args.run / "answer_generator.json"
    generator.save(generator_checkpoint_path)
    return {
        "architecture": "transformer-guided-answer-generator",
        "checkpoint": str(generator_checkpoint_path),
        "history": str(generator_history_path),
        "steps": args.generator_steps,
        "training_examples": len(generator_training_pool),
        "learning_rate": args.generator_learning_rate,
        "generator_eval_every": args.generator_eval_every,
        "max_answer_chars": args.generator_max_answer_chars,
        "transformer_top_k": args.generator_transformer_top_k,
        "labels": len(generator.config.labels),
        "features": len(generator.config.features),
        "baseline": generator_baseline,
        "final": last_generator_snapshot,
        "uses_answer_candidates": False,
        "pretrained_weights": False,
        "pretrained_tokenizer": False,
        "external_embeddings": False,
        "training_data": TRAINING_DATA_DESCRIPTION,
    }
