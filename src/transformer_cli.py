"""Command-line parsing for transformer training and evaluation."""

from __future__ import annotations

import argparse
import json
from typing import Any, Callable

from transformer_cli_answer_parser import add_answer_train_parser
from transformer_cli_eval_parser import add_eval_parser
from transformer_cli_train_parser import add_train_parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_train_parser(subparsers)
    add_eval_parser(subparsers)
    add_answer_train_parser(subparsers)
    return parser.parse_args(argv)


def run_transformer_cli(
    argv: list[str] | None,
    *,
    train_transformer: Callable[[argparse.Namespace], dict[str, Any]],
    eval_transformer: Callable[[argparse.Namespace], dict[str, Any]],
    train_transformer_answers: Callable[[argparse.Namespace], dict[str, Any]],
) -> int:
    args = parse_args(argv)
    if args.command == "train":
        train_transformer(args)
        return 0
    if args.command == "eval":
        result = eval_transformer(args)
        print(json.dumps(result["evals"], indent=2, sort_keys=True))
        return 0
    if args.command == "answer-train":
        train_transformer_answers(args)
        return 0
    raise ValueError(f"unknown command {args.command!r}")
