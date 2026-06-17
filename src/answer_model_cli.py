"""CLI parsing and dispatch for the answer model."""

from __future__ import annotations

import argparse
from pathlib import Path

from answer_model_commands import eval_model
from answer_model_constants import DEFAULT_RUN_DIR, DEFAULT_TRAIN_TEXT
from answer_model_training import train_model
from curriculum import DEFAULT_CORPUS_DIR


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--train-text", type=Path, default=DEFAULT_TRAIN_TEXT)
    train.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    train.add_argument("--run", type=Path, default=DEFAULT_RUN_DIR)
    train.add_argument("--steps", type=int, default=2000)
    train.add_argument("--learning-rate", type=float, default=0.08)
    train.add_argument("--eval-every", type=int, default=200)
    train.add_argument("--seed", type=int, default=7)

    evaluate = subparsers.add_parser("eval")
    evaluate.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_RUN_DIR / "answer_model.json",
    )
    evaluate.add_argument("--json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "train":
        train_model(args)
        return 0
    if args.command == "eval":
        eval_model(args)
        return 0
    raise AssertionError(args.command)
