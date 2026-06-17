"""Data loading and lesson artifact writers for the answer model."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from answer_examples import AnswerExample, examples_from_sources
from curriculum import read_json


def load_training_examples(train_text_path: Path, corpus_dir: Path) -> list[AnswerExample]:
    grammar = read_json(corpus_dir / "grammar.json")
    glossary = read_json(corpus_dir / "glossary.json")
    train_text = train_text_path.read_text(encoding="utf-8")
    return examples_from_sources(train_text, grammar, glossary)


def write_lessons(examples: list[AnswerExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(asdict(example), sort_keys=True) + "\n")
