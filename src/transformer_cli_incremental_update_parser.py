"""Incremental update subcommand parser for the transformer CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from transformer_cli_shared_options import add_generation_sampling_options


def add_incremental_update_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "incremental-update",
        help="guard and promote an incremental transformer checkpoint",
    )
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--accepted-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--new-lesson-probe", action="append", type=Path, required=True)
    parser.add_argument("--regression-probe", action="append", type=Path, required=True)
    parser.add_argument("--nll-tolerance", type=float, default=0.0)
    add_generation_sampling_options(parser)
