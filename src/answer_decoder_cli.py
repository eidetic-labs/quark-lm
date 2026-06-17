"""CLI parsing and dispatch for the answer decoder."""

from __future__ import annotations

import argparse
from pathlib import Path

from answer_decoder_commands import eval_decoder
from answer_decoder_constants import DEFAULT_DECODER_RUN_DIR
from answer_decoder_training import train_decoder
from answer_model import DEFAULT_CORPUS_DIR, DEFAULT_TRAIN_TEXT


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--train-text", type=Path, default=DEFAULT_TRAIN_TEXT)
    train.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    train.add_argument("--run", type=Path, default=DEFAULT_DECODER_RUN_DIR)
    train.add_argument("--steps", type=int, default=2200)
    train.add_argument("--learning-rate", type=float, default=0.04)
    train.add_argument("--eval-every", type=int, default=550)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--max-answer-chars", type=int, default=64)

    evaluate = subparsers.add_parser("eval")
    evaluate.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_DECODER_RUN_DIR / "answer_decoder.json",
    )
    evaluate.add_argument("--json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "train":
        train_decoder(args)
        return 0
    if args.command == "eval":
        eval_decoder(args)
        return 0
    raise AssertionError(args.command)
