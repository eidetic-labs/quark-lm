"""Generate the admitted nursery corpus from ledgered source files."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from curriculum_sections import (
    glossary_lines,
    self_knowledge_lines,
    sentence_lines,
    story_lines,
    unknown_lesson_lines,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_DIR = PROJECT_DIR / "corpus"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "build"


@dataclass(frozen=True)
class Curriculum:
    train_text: str
    valid_text: str
    manifest: dict[str, Any]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_curriculum(corpus_dir: Path = DEFAULT_CORPUS_DIR, seed: int = 7) -> Curriculum:
    ledger = read_json(corpus_dir / "ledger.json")
    glossary = read_json(corpus_dir / "glossary.json")
    grammar = read_json(corpus_dir / "grammar.json")
    admitted_facts = read_jsonl(corpus_dir / "admissions.jsonl")
    entries = glossary["entries"]
    rng = random.Random(seed)

    train_sentences, valid_sentences = sentence_lines(entries, grammar, rng)
    train_sections = [
        *glossary_lines(entries),
        "",
        *train_sentences,
        "",
        *story_lines(grammar, admitted_facts),
        "",
        *self_knowledge_lines(grammar),
        "",
        *unknown_lesson_lines(grammar),
    ]
    valid_sections = [
        *valid_sentences,
        "",
        "validation closed world lessons:",
        "question: what is not in this world?",
        "answer: unknown.",
    ]
    train_text = "\n".join(train_sections).strip() + "\n"
    valid_text = "\n".join(valid_sections).strip() + "\n"
    manifest = {
        "seed": seed,
        "ledger_version": ledger["version"],
        "glossary_entries": len(entries),
        "sentence_templates": len(grammar["sentence_templates"]),
        "story_facts": len(grammar["story_facts"]),
        "admitted_facts": len(admitted_facts),
        "qa_lesson_facts": len(grammar.get("qa_lesson_ids", [])),
        "heldout_probe_facts": len(grammar.get("heldout_probe_ids", [])),
        "unknown_facts": len(grammar["unknown_facts"]),
        "unknown_owner_objects": len(grammar.get("unknown_owner_objects", [])),
        "self_facts": len(grammar.get("self_facts", [])),
        "learning_rules": len(grammar.get("learning_rules", [])),
        "train_chars": len(train_text),
        "valid_chars": len(valid_text),
        "sources": ledger["sources"],
    }
    return Curriculum(train_text=train_text, valid_text=valid_text, manifest=manifest)


def write_curriculum(curriculum: Curriculum, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train.txt").write_text(curriculum.train_text, encoding="utf-8")
    (output_dir / "valid.txt").write_text(curriculum.valid_text, encoding="utf-8")
    write_json(output_dir / "manifest.json", curriculum.manifest)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    curriculum = build_curriculum(args.corpus_dir, args.seed)
    write_curriculum(curriculum, args.output)
    print(f"wrote {args.output / 'train.txt'} ({len(curriculum.train_text)} chars)")
    print(f"wrote {args.output / 'valid.txt'} ({len(curriculum.valid_text)} chars)")
    print(f"wrote {args.output / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
