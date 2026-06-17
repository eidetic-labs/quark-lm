"""Artifact writers for answer decoder training runs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from answer_model import AnswerExample


def write_lessons(examples: list[AnswerExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(asdict(example), sort_keys=True) + "\n")
