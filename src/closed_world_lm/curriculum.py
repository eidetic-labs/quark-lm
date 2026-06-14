"""Generate the admitted nursery corpus from ledgered source files."""

from __future__ import annotations

import argparse
import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[2]
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


def entries_for_tag(entries: list[dict[str, Any]], tag: str) -> list[str]:
    words = [entry["word"] for entry in entries if tag in entry.get("tags", [])]
    return sorted(dict.fromkeys(words))


def slot_values(
    entries: list[dict[str, Any]],
    grammar: dict[str, Any],
    slot_name: str,
    tag_name: str,
) -> list[str]:
    overrides = grammar.get("slot_overrides", {})
    if slot_name in overrides:
        return list(overrides[slot_name])
    values = entries_for_tag(entries, tag_name)
    if not values:
        raise ValueError(f"slot {slot_name!r} with tag {tag_name!r} has no values")
    return values


def fields_in_template(template: str) -> list[str]:
    fields: list[str] = []
    for _, field_name, _, _ in Formatter().parse(template):
        if field_name:
            fields.append(field_name)
    return fields


def render_limited(
    template: str,
    slots: dict[str, str],
    entries: list[dict[str, Any]],
    grammar: dict[str, Any],
    limit: int,
    rng: random.Random,
) -> list[str]:
    field_names = fields_in_template(template)
    value_lists = [
        slot_values(entries, grammar, field_name, slots[field_name]) for field_name in field_names
    ]
    combinations = list(itertools.product(*value_lists))
    rng.shuffle(combinations)
    rendered: list[str] = []
    for combo in combinations[:limit]:
        values = dict(zip(field_names, combo, strict=True))
        rendered.append(template.format(**values))
    return sorted(rendered)


def glossary_lines(entries: list[dict[str, Any]]) -> list[str]:
    lines = ["glossary:"]
    for entry in sorted(entries, key=lambda item: item["word"]):
        lines.append(f"{entry['word']}: {entry['definition']}.")
    return lines


def sentence_lines(
    entries: list[dict[str, Any]],
    grammar: dict[str, Any],
    rng: random.Random,
) -> tuple[list[str], list[str]]:
    train: list[str] = ["sentences:"]
    valid: list[str] = ["validation sentences:"]
    for item in grammar["sentence_templates"]:
        lines = render_limited(
            item["template"],
            item["slots"],
            entries,
            grammar,
            int(item["limit"]),
            rng,
        )
        for index, line in enumerate(lines):
            if index % 5 == 0:
                valid.append(line)
            else:
                train.append(line)
    return train, valid


def story_lines(grammar: dict[str, Any], admitted_facts: list[dict[str, Any]]) -> list[str]:
    lines = ["stories:"]
    qa_lesson_ids = set(grammar.get("qa_lesson_ids", []))
    admitted_ids = {fact["id"] for fact in admitted_facts}
    for fact in [*grammar["story_facts"], *admitted_facts]:
        person = fact["person"]
        obj = fact["object"]
        color = fact["color"]
        relation = fact["relation"]
        container = fact["container"]
        if fact["id"] in admitted_ids:
            lines.extend(
                [
                    f"event: I learned something new: {person}'s {obj} is {relation} the {container}.",
                    f"event: now {person}'s {obj} is part of my training data.",
                ]
            )
        lines.extend(
            [
                f"story: {person} has a {color} {obj}.",
                f"{person} puts the {obj} {relation} the {container}.",
                f"fact: {person}'s {obj} is {relation} the {container}.",
                f"fact: {person}'s {obj} color is {color}.",
                f"fact: the {obj} belongs to {person}.",
            ]
        )
        if fact["id"] in qa_lesson_ids:
            lines.extend(
                [
                    f"question: where is {person}'s {obj}?",
                    f"answer: {relation} the {container}.",
                    f"question: what color is {person}'s {obj}?",
                    f"answer: {color}.",
                ]
            )
        else:
            lines.extend(
                [
                    f"lesson: {person}'s {obj} place answer is {relation} the {container}.",
                    f"lesson: {person}'s {obj} color answer is {color}.",
                ]
            )
    return lines


def self_knowledge_lines(grammar: dict[str, Any]) -> list[str]:
    lines = ["self knowledge:"]
    for fact in grammar.get("self_facts", []):
        lines.append(f"fact: self {fact['slot']} is {fact['answer']}.")
    for rule in grammar.get("learning_rules", []):
        lines.append(f"fact: learning {rule['slot']} means {rule['answer']}.")
    lines.extend(
        [
            "if I learn something new, it must be admitted before training.",
            "when new data is admitted, training may update weights.",
        ]
    )
    return lines


def unknown_lesson_lines(grammar: dict[str, Any]) -> list[str]:
    lines = ["closed world lessons:"]
    for fact in grammar["unknown_facts"]:
        person = fact["person"]
        obj = fact["object"]
        lines.extend(
            [
                f"question: where is {person}'s {obj}?",
                "answer: unknown.",
            ]
        )
    lines.extend(
        [
            "if a fact is not in this world, the answer is unknown.",
            "unknown means not known in this world.",
        ]
    )
    return lines


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
