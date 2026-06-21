"""Section renderers for the admitted nursery curriculum."""

from __future__ import annotations

import itertools
import random
from string import Formatter
from typing import Any


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
        slot_values(entries, grammar, field_name, slots[field_name])
        for field_name in field_names
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
    withheld_ids = set(grammar.get("withheld_fact_ids", []))
    admitted_ids = {fact["id"] for fact in admitted_facts}
    for fact in [*grammar["story_facts"], *admitted_facts]:
        # Withheld facts are fully excluded from the corpus prose (and therefore from
        # the parsed training examples + the oracle), so the model is never admitted
        # them and must abstain -- the genuine fact-level closed-world boundary.
        if fact["id"] in withheld_ids:
            continue
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
