"""Eval subcommand parser for the transformer CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from transformer_cli_shared_options import add_generation_sampling_options
from transformer_paths import DEFAULT_CHECKPOINT


def add_eval_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    eval_parser = subparsers.add_parser("eval", help="evaluate the tiny transformer")
    eval_parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    eval_parser.add_argument("--max-new-chars", type=int, default=24)
    eval_parser.add_argument("--json", type=Path, default=None)
    eval_parser.add_argument("--samples-jsonl", type=Path, default=None)
    add_generation_sampling_options(eval_parser)
    eval_parser.add_argument(
        "--probe",
        action="append",
        type=Path,
        default=None,
        help="JSONL probe file. Defaults to qa, unknowns, heldout, and paraphrases.",
    )
