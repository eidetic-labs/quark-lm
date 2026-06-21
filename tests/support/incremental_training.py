"""Fixtures for transformer incremental training and guarded update tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from support.commands import parse_args
from transformer_char_model import train_transformer


def write_text(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def write_jsonl_records(path: Path, records: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def train_args(
    *,
    corpus: Path,
    valid: Path,
    run: Path,
    steps: int,
    resume_checkpoint: Path | None = None,
) -> argparse.Namespace:
    argv = [
        "train",
        "--corpus",
        str(corpus),
        "--valid",
        str(valid),
        "--run",
        str(run),
        "--steps",
        str(steps),
        "--learning-rate",
        "0.08",
        # Pin sgd so these guarded-update fixtures stay deterministic and
        # decoupled from the default optimizer (now adamw).
        "--optimizer",
        "sgd",
        "--eval-every",
        "0",
        "--valid-limit",
        "128",
        "--context-size",
        "8",
        "--embedding-dim",
        "4",
        "--feedforward-dim",
        "8",
        "--seed",
        "1",
    ]
    if resume_checkpoint is not None:
        argv.extend(["--resume-checkpoint", str(resume_checkpoint)])
    return parse_args(argv)


def train_incremental_candidate(
    root: Path,
    *,
    old_repeats: int,
    new_repeats: int,
    candidate_steps: int,
) -> tuple[dict[str, object], dict[str, object]]:
    base_text = "q:\na: no\n" * old_repeats
    expanded_text = base_text + ("q2:\na: ok!\n" * new_repeats)
    base_corpus = write_text(root / "base.txt", base_text)
    base_valid = write_text(root / "base_valid.txt", base_text)
    expanded_corpus = write_text(root / "expanded.txt", expanded_text)
    expanded_valid = write_text(root / "expanded_valid.txt", expanded_text)
    base_metrics = train_transformer(
        train_args(
            corpus=base_corpus,
            valid=base_valid,
            run=root / "base_run",
            steps=640,
        )
    )
    candidate_metrics = train_transformer(
        train_args(
            corpus=expanded_corpus,
            valid=expanded_valid,
            run=root / "candidate_run",
            steps=candidate_steps,
            resume_checkpoint=Path(str(base_metrics["checkpoint"])),
        )
    )
    return base_metrics, candidate_metrics


def record(record_id: str, prompt: str, target: str) -> dict[str, str]:
    return {"id": record_id, "prompt": prompt, "target": target}
