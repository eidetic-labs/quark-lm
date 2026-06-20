"""Answer-sweep subcommand parser for controlled transformer screens."""

from __future__ import annotations

import argparse
from pathlib import Path

from transformer_cli_answer_parser import add_answer_train_arguments


def add_answer_sweep_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    sweep_parser = subparsers.add_parser(
        "answer-sweep",
        help="run controlled answer-training trials across declared axes",
    )
    add_answer_train_arguments(sweep_parser)
    sweep_parser.add_argument(
        "--sweep-axis",
        action="append",
        default=None,
        help="Controlled axis formatted as name=value[,value]. May be repeated.",
    )
    sweep_parser.add_argument(
        "--sweep-report",
        type=Path,
        default=None,
        help="Path for the combined sweep report. Defaults to RUN/sweep_report.json.",
    )
    sweep_parser.add_argument(
        "--sweep-frontier-metrics",
        type=Path,
        default=None,
        help=(
            "Optional frontier metrics JSON used to compare each trial against "
            "the best proven branch-diversity evidence."
        ),
    )
    sweep_parser.add_argument(
        "--sweep-max-trials",
        type=int,
        default=16,
        help="Maximum cartesian-product trials allowed for one sweep run.",
    )
    sweep_parser.add_argument(
        "--sweep-dry-run",
        action="store_true",
        help="Write the sweep report without executing trial training.",
    )
