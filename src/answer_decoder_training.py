"""Training workflow for answer decoder checkpoints."""

from __future__ import annotations

import argparse
import json
import random
from typing import Any

from answer_decoder_artifacts import write_lessons
from answer_decoder_builder import build_decoder
from answer_decoder_evaluation import summarize_eval
from answer_decoder_pool import decoder_training_pool
from answer_model import DEFAULT_EVALS, load_training_examples
from probes import read_jsonl


def train_decoder(args: argparse.Namespace) -> dict[str, Any]:
    examples = load_training_examples(args.train_text, args.corpus_dir)
    training_pool = decoder_training_pool(examples)
    model = build_decoder(examples, args.seed, args.max_answer_chars)
    rng = random.Random(args.seed)
    args.run.mkdir(parents=True, exist_ok=True)
    history_path = args.run / "decoder_metrics.jsonl"
    lessons_path = args.run / "decoder_lessons.jsonl"
    write_lessons(examples, lessons_path)

    def snapshot(step: int, train_loss: float | None) -> dict[str, Any]:
        result = {
            "step": step,
            "train_loss": train_loss,
            "evals": {
                path.stem: summarize_eval(model, read_jsonl(path))
                for path in DEFAULT_EVALS
            },
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
        return result

    baseline = snapshot(0, None)
    running_loss = 0.0
    last_snapshot = baseline
    last_snapshot_step = 0
    pool_order = training_pool[:]
    rng.shuffle(pool_order)
    pool_index = 0
    for step in range(1, args.steps + 1):
        if pool_index == len(pool_order):
            rng.shuffle(pool_order)
            pool_index = 0
        example = pool_order[pool_index]
        pool_index += 1
        running_loss += model.train_example(example, args.learning_rate)
        if step % args.eval_every == 0:
            train_loss = running_loss / args.eval_every
            last_snapshot = snapshot(step, train_loss)
            last_snapshot_step = step
            print(f"step={step} train_loss={train_loss:.4f}")
            running_loss = 0.0

    if last_snapshot_step != args.steps:
        last_snapshot = snapshot(args.steps, None)

    checkpoint_path = args.run / "answer_decoder.json"
    model.save(checkpoint_path)
    metrics = {
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
        "lessons": str(lessons_path),
        "steps": args.steps,
        "examples": len(examples),
        "training_examples": len(training_pool),
        "labels": len(model.config.labels),
        "features": len(model.config.features),
        "baseline": baseline,
        "final": last_snapshot,
    }
    with (args.run / "decoder_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"saved {checkpoint_path}")
    return metrics
