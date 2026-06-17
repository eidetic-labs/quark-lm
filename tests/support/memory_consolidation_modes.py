"""Memory-consolidation mode fixtures used by transformer tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from support.commands import parse_args, train_transformer_answers


def write_memory_consolidation_source_plan(
    run_root: str | Path,
    *,
    collapsed_profiles: list[str],
    top_priority_profiles: list[str],
    profile_priorities: list[dict[str, Any]],
) -> Path:
    source_plan = Path(run_root) / "source_memory_consolidation_plan.json"
    with source_plan.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "kind": "memory_consolidation_plan",
                "summary": {
                    "collapsed_memory_backed_profiles": collapsed_profiles,
                    "top_priority_profiles": top_priority_profiles,
                    "memory_backed_failed_profiles": 9,
                    "retrieval_exact_rate": 1.0,
                },
                "profile_priorities": profile_priorities,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    return source_plan


def train_memory_consolidation_mode_screen(
    run_root: str | Path,
    *,
    run_name: str,
    direct_answer_mode: str,
    source_plan: str | Path,
    max_profiles: int = 3,
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
            "--memory-consolidation-source-plan",
            str(source_plan),
            "--memory-consolidation-max-profiles",
            str(max_profiles),
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


__all__ = [
    "train_memory_consolidation_mode_screen",
    "write_memory_consolidation_source_plan",
]
