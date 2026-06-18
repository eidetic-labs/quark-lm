"""Train subcommand parser for the transformer CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from curriculum import DEFAULT_OUTPUT_DIR
from transformer_cli_shared_options import (
    add_architecture_options,
    add_optimizer_options,
    add_tokenizer_options,
)
from transformer_paths import DEFAULT_RUN_DIR


def add_train_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    train_parser = subparsers.add_parser("train", help="train the tiny transformer")
    train_parser.add_argument("--corpus", type=Path, default=DEFAULT_OUTPUT_DIR / "train.txt")
    train_parser.add_argument("--valid", type=Path, default=DEFAULT_OUTPUT_DIR / "valid.txt")
    train_parser.add_argument("--run", type=Path, default=DEFAULT_RUN_DIR)
    train_parser.add_argument("--steps", type=int, default=80)
    train_parser.add_argument("--learning-rate", type=float, default=0.03)
    add_architecture_options(train_parser)
    add_tokenizer_options(train_parser)
    train_parser.add_argument("--seed", type=int, default=17)
    train_parser.add_argument("--eval-every", type=int, default=20)
    train_parser.add_argument("--valid-limit", type=int, default=256)
    add_optimizer_options(train_parser)
    train_parser.add_argument("--resume-checkpoint", type=Path, default=None)
    train_parser.add_argument("--resume-optimizer", type=Path, default=None)
