"""Baseline-floor mode command fixtures used by transformer tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from support.commands import parse_args, train_transformer_answers


def train_baseline_floor_mode_screen(
    run_root: str | Path,
    run_name: str,
    direct_answer_mode: str,
) -> dict[str, Any]:
    args = parse_args(
        [
            "answer-train",
            "--run",
            str(Path(run_root) / run_name),
            "--steps",
            "0",
            "--eval-every",
            "0",
            "--candidate-scope",
            "eval",
            "--direct-answer-steps",
            "1",
            "--direct-answer-eval-every",
            "1",
            "--direct-answer-mode",
            direct_answer_mode,
            "--direct-answer-snapshot-mode",
            "branch-only",
            "--direct-answer-branch-batch-size",
            "2",
            "--direct-answer-hard-negatives",
            "1",
            "--skip-post-direct-snapshot",
            "--embedding-dim",
            "2",
            "--feedforward-dim",
            "4",
            "--context-size",
            "80",
        ]
    )
    return train_transformer_answers(args)


__all__ = ["train_baseline_floor_mode_screen"]
