"""Deterministic closed-world responses learned from admitted corpus facts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from curriculum import DEFAULT_OUTPUT_DIR


DEFAULT_TRAIN_TEXT = DEFAULT_OUTPUT_DIR / "train.txt"

PLACE_FACT_RE = re.compile(r"^fact: (?P<person>[a-z]+)'s (?P<object>[a-z]+) is (?P<place>.+)\.$")
COLOR_FACT_RE = re.compile(
    r"^fact: (?P<person>[a-z]+)'s (?P<object>[a-z]+) color is (?P<color>[a-z]+)\.$"
)
OWNER_FACT_RE = re.compile(r"^fact: the (?P<object>[a-z]+) belongs to (?P<person>[a-z]+)\.$")
SELF_FACT_RE = re.compile(r"^fact: self (?P<slot>[a-z_]+) is (?P<answer>.+)\.$")
LEARNING_FACT_RE = re.compile(
    r"^fact: learning (?P<slot>[a-z_]+) means (?P<answer>.+)\.$"
)
GLOSSARY_LINE_RE = re.compile(r"^(?P<word>[a-z]+): (?P<definition>.+)\.$")
WHERE_QUESTION_RE = re.compile(r"question: where is (?P<person>[a-z]+)'s (?P<object>[a-z]+)\?")
COLOR_QUESTION_RE = re.compile(
    r"question: what color is (?P<person>[a-z]+)'s (?P<object>[a-z]+)\?"
)
OWNER_QUESTION_RE = re.compile(r"question: who has the (?P<object>[a-z]+)\?")
TRAINING_DATA_QUESTION_RE = re.compile(
    r"question: is (?P<person>[a-z]+)'s (?P<object>[a-z]+) part of your training data\?"
)
TRAINING_DATA_TAG_RE = re.compile(r"training data: (?P<person>[a-z]+) (?P<object>[a-z]+)")
PLACE_ASK_RE = re.compile(r"ask: place for (?P<person>[a-z]+) (?P<object>[a-z]+)")
COLOR_ASK_RE = re.compile(r"ask: color for (?P<person>[a-z]+) (?P<object>[a-z]+)")
OWNER_ASK_RE = re.compile(r"ask: owner for (?P<object>[a-z]+)")
PLACE_TELL_RE = re.compile(r"tell me the place of (?P<person>[a-z]+) (?P<object>[a-z]+)")
COLOR_BELONGS_RE = re.compile(r"which color belongs to (?P<person>[a-z]+) (?P<object>[a-z]+)")
OWNER_BELONGS_RE = re.compile(r"which person has (?P<object>[a-z]+)")
GLOSSARY_MEANING_RE = re.compile(r"question: what does (?P<word>[a-z]+) mean\?")
GLOSSARY_DEFINE_RE = re.compile(r"define (?P<word>[a-z]+)")
SELF_PROMPT_PATTERNS = [
    ("kind", re.compile(r"question: what are you\?")),
    ("dataset", re.compile(r"question: what is your dataset\?")),
    ("pretrained_weights", re.compile(r"question: do you use pretrained weights\?")),
    (
        "unknown_policy",
        re.compile(r"question: what do you say when a fact is outside your corpus\?"),
    ),
    ("improvement_method", re.compile(r"question: how do you improve\?")),
    ("diagnosis_source", re.compile(r"question: what source guides your self-diagnosis\?")),
    (
        "external_model_shaping",
        re.compile(r"question: does an external model shape your self-diagnosis\?"),
    ),
]
LEARNING_PROMPT_PATTERNS = [
    ("new_data", re.compile(r"question: what happens when you learn something new\?")),
    ("admission", re.compile(r"question: when is something learned\?")),
    (
        "weight_update",
        re.compile(r"question: what changes after new training data is admitted\?"),
    ),
    ("repair_action", re.compile(r"question: how is the next repair action chosen\?")),
]


@dataclass(frozen=True)
class FactRecord:
    place: str | None = None
    color: str | None = None
    owner: str | None = None


class CorpusResponder:
    def __init__(
        self,
        facts: dict[tuple[str, str], FactRecord],
        self_facts: dict[str, str] | None = None,
        learning_rules: dict[str, str] | None = None,
        glossary: dict[str, str] | None = None,
    ) -> None:
        self.facts = facts
        self.self_facts = self_facts or {}
        self.learning_rules = learning_rules or {}
        self.glossary = glossary or {}

    @classmethod
    def train_from_text(cls, text: str) -> "CorpusResponder":
        mutable: dict[tuple[str, str], dict[str, str]] = {}
        self_facts: dict[str, str] = {}
        learning_rules: dict[str, str] = {}
        glossary: dict[str, str] = {}
        in_glossary = False
        for line in text.splitlines():
            if line == "glossary:":
                in_glossary = True
                continue
            if in_glossary:
                if not line.strip():
                    in_glossary = False
                    continue
                glossary_match = GLOSSARY_LINE_RE.match(line)
                if glossary_match:
                    glossary[glossary_match["word"]] = glossary_match["definition"]
                    continue

            place_match = PLACE_FACT_RE.match(line)
            if place_match:
                key = (place_match["person"], place_match["object"])
                mutable.setdefault(key, {})["place"] = place_match["place"]
                continue
            color_match = COLOR_FACT_RE.match(line)
            if color_match:
                key = (color_match["person"], color_match["object"])
                mutable.setdefault(key, {})["color"] = color_match["color"]
                continue
            owner_match = OWNER_FACT_RE.match(line)
            if owner_match:
                key = (owner_match["person"], owner_match["object"])
                mutable.setdefault(key, {})["owner"] = owner_match["person"]
                continue
            self_match = SELF_FACT_RE.match(line)
            if self_match:
                self_facts[self_match["slot"]] = self_match["answer"]
                continue
            learning_match = LEARNING_FACT_RE.match(line)
            if learning_match:
                learning_rules[learning_match["slot"]] = learning_match["answer"]

        facts = {
            key: FactRecord(
                place=value.get("place"),
                color=value.get("color"),
                owner=value.get("owner"),
            )
            for key, value in mutable.items()
        }
        return cls(facts, self_facts, learning_rules, glossary)

    @classmethod
    def load_train_text(cls, path: Path = DEFAULT_TRAIN_TEXT) -> "CorpusResponder":
        return cls.train_from_text(path.read_text(encoding="utf-8"))

    def answer_prompt(self, prompt: str) -> str:
        where_match = (
            WHERE_QUESTION_RE.search(prompt)
            or PLACE_ASK_RE.search(prompt)
            or PLACE_TELL_RE.search(prompt)
        )
        if where_match:
            fact = self.facts.get((where_match["person"], where_match["object"]))
            if fact and fact.place:
                return f" {fact.place}."
            return " unknown."

        color_match = (
            COLOR_QUESTION_RE.search(prompt)
            or COLOR_ASK_RE.search(prompt)
            or COLOR_BELONGS_RE.search(prompt)
        )
        if color_match:
            fact = self.facts.get((color_match["person"], color_match["object"]))
            if fact and fact.color:
                return f" {fact.color}."
            return " unknown."

        owner_match = (
            OWNER_QUESTION_RE.search(prompt)
            or OWNER_ASK_RE.search(prompt)
            or OWNER_BELONGS_RE.search(prompt)
        )
        if owner_match:
            obj = owner_match["object"]
            for (person, fact_obj), fact in self.facts.items():
                if fact_obj == obj and fact.owner:
                    return f" {person}."
            return " unknown."

        training_data_match = (
            TRAINING_DATA_QUESTION_RE.search(prompt)
            or TRAINING_DATA_TAG_RE.search(prompt)
        )
        if training_data_match:
            key = (training_data_match["person"], training_data_match["object"])
            return " yes." if key in self.facts else " no."

        glossary_match = GLOSSARY_MEANING_RE.search(prompt) or GLOSSARY_DEFINE_RE.search(
            prompt
        )
        if glossary_match:
            definition = self.glossary.get(glossary_match["word"])
            return f" {definition}." if definition else " unknown."

        for slot, pattern in SELF_PROMPT_PATTERNS:
            if pattern.search(prompt):
                answer = self.self_facts.get(slot)
                return f" {answer}." if answer else " unknown."

        for slot, pattern in LEARNING_PROMPT_PATTERNS:
            if pattern.search(prompt):
                answer = self.learning_rules.get(slot)
                return f" {answer}." if answer else " unknown."

        return " unknown."

    def evaluate(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        scored = []
        for record in records:
            answer = self.answer_prompt(record["prompt"])
            scored.append(
                {
                    "id": record["id"],
                    "target": record["target"],
                    "answer": answer,
                    "exact_match": answer == record["target"],
                }
            )
        exact = sum(1 for record in scored if record["exact_match"])
        return {
            "count": len(scored),
            "exact": exact,
            "exact_rate": exact / len(scored) if scored else 0.0,
            "records": scored,
        }
